from flask import Flask, url_for, redirect, render_template, request, session, Response
import pymysql, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non‑interactive backend for saving plots
import matplotlib.pyplot as plt
import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'admin'

# ─── Database Connection ───────────────────────────────────────────────────────
mydb = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    port=3306,
    database='aerial'
)
mycursor = mydb.cursor()

def executionquery(query, values):
    mycursor.execute(query, values)
    mydb.commit()

def retrivequery1(query, values):
    mycursor.execute(query, values)
    return mycursor.fetchall()

def retrivequery2(query):
    mycursor.execute(query)
    return mycursor.fetchall()


# ─── Database Schema Initialisation (with severity & prognosis) ───────────────
def init_db():
    """Ensure analysis_history table has all required columns."""
    # Check if table exists
    check_table = """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'aerial' AND table_name = 'analysis_history'
    """
    mycursor.execute(check_table)
    table_exists = mycursor.fetchone()[0] > 0

    if not table_exists:
        # Create new table with all columns
        create_table = """
            CREATE TABLE analysis_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_email VARCHAR(255) NOT NULL,
                filename VARCHAR(255) NOT NULL,
                image_path VARCHAR(500) NOT NULL,
                xai_path VARCHAR(500),
                predicted_class VARCHAR(100),
                confidence VARCHAR(50),
                top1_class VARCHAR(100),
                top2_class VARCHAR(100),
                top3_class VARCHAR(100),
                top1_conf FLOAT,
                top2_conf FLOAT,
                top3_conf FLOAT,
                severity VARCHAR(50),
                prognosis TEXT,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_email) REFERENCES users(email) ON DELETE CASCADE
            )
        """
        mycursor.execute(create_table)
        mydb.commit()
        print("Created new analysis_history table with severity & prognosis.")
    else:
        # Table exists – add missing columns if needed
        columns_to_add = [
            ('severity', 'VARCHAR(50)'),
            ('prognosis', 'TEXT')
        ]
        for col_name, col_type in columns_to_add:
            mycursor.execute(f"""
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = 'aerial'
                  AND table_name = 'analysis_history'
                  AND column_name = '{col_name}'
            """)
            if mycursor.fetchone()[0] == 0:
                alter_sql = f"ALTER TABLE analysis_history ADD COLUMN {col_name} {col_type}"
                mycursor.execute(alter_sql)
                mydb.commit()
                print(f"Added column '{col_name}' to analysis_history.")

# Call init_db after establishing connection
init_db()


# ─── Auth Routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email      = request.form['email']
        password   = request.form['password']
        c_password = request.form['c_password']

        if password != c_password:
            return render_template('register.html', message="Confirm password does not match!")

        existing = [row[0] for row in retrivequery2("SELECT UPPER(email) FROM users")]
        if email.upper() in existing:
            return render_template('register.html', message="This email ID already exists!")

        executionquery("INSERT INTO users (email, password) VALUES (%s, %s)", (email, password))
        return render_template('login.html', message="Successfully Registered!")

    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form['email']
        password = request.form['password']

        existing = [row[0] for row in retrivequery2("SELECT UPPER(email) FROM users")]
        if email.upper() not in existing:
            return render_template('login.html', message="This email ID does not exist!")

        stored_pw = retrivequery1("SELECT UPPER(password) FROM users WHERE email = %s", (email,))
        if password.upper() != stored_pw[0][0]:
            return render_template('login.html', message="Invalid Password!!")

        user_data = retrivequery1("SELECT id FROM users WHERE email = %s", (email,))
        user_id = user_data[0][0]
        session['user_email'] = email
        session['user_id'] = user_id
        return redirect(url_for('home'))

    return render_template('login.html')


@app.route('/home')
def home():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')


@app.route('/about')
def about():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('about.html')


@app.route('/accuracy')
def accuracy():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('accuracy.html')


@app.route('/logout')
def logout():
    session.pop('user_email', None)
    session.pop('user_id', None)
    return redirect(url_for('index'))


# ───────────────────────────────────────────────────────────────────────────────
# QCNN Model Definition (must match training)
# ───────────────────────────────────────────────────────────────────────────────
class QCNN(nn.Module):
    def __init__(self, Cap=2.0):
        super(QCNN, self).__init__()
        self.Cap = Cap
        self.conv1 = nn.Conv2d(3, 5, kernel_size=7)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=7)
        self.fc1 = nn.Linear(449440, 300)   # 224->218->212, then flatten: 212*212*10 ≈ 449440
        self.fc2 = nn.Linear(300, 50)
        self.fc3 = nn.Linear(50, 4)          # 4 classes

    def activation_function_CapReLu(self, x):
        return torch.clamp(x, min=0, max=self.Cap)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation_function_CapReLu(x)
        x = self.conv2(x)
        x = self.activation_function_CapReLu(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


# ─── XAI Analyzer (adapted from the standalone script) ─────────────────────────
class XQAI_Analyzer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.model.eval()
        self.device = device
        self.activations = {}
        self.gradients = {}
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        self.model.conv1.register_forward_hook(forward_hook('conv1'))
        self.model.conv1.register_backward_hook(backward_hook('conv1'))
        self.model.conv2.register_forward_hook(forward_hook('conv2'))
        self.model.conv2.register_backward_hook(backward_hook('conv2'))

    def quantum_saliency_map(self, input_image, target_class=None):
        input_image = input_image.unsqueeze(0).requires_grad_()
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()
        saliency = input_image.grad[0].abs().max(dim=0)[0].detach().cpu().numpy()
        return saliency

    def quantum_cam(self, input_image, target_class=None):
        input_image = input_image.unsqueeze(0).requires_grad_()
        features = []
        def hook_fn(module, input, output):
            features.append(output)
        handle = self.model.conv2.register_forward_hook(hook_fn)
        output = self.model(input_image)
        handle.remove()
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        fc_weights = self.model.fc3.weight[target_class].detach().to(self.device)
        feature_maps = features[0][0]
        cam = torch.zeros(feature_maps.shape[1:], device=self.device)
        for i, w in enumerate(fc_weights[:len(feature_maps)]):
            cam += w * feature_maps[i]
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def lrp_analysis(self, input_image, target_class=None):
        input_image = input_image.unsqueeze(0).requires_grad_()
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        relevance = torch.zeros_like(output)
        relevance[0, target_class] = output[0, target_class]
        self.model.zero_grad()
        output.backward(gradient=relevance)
        relevance_maps = {}
        relevance_maps['input'] = (input_image.grad[0] * input_image[0]).detach().cpu().numpy()
        for name, grad in self.gradients.items():
            if name in self.activations:
                relevance_maps[name] = (grad * self.activations[name])[0].detach().cpu().numpy()
        return relevance_maps

    def visualize_quantum_states(self, input_image, layer='conv1'):
        with torch.no_grad():
            _ = self.model(input_image.unsqueeze(0))
            if layer in self.activations:
                act = self.activations[layer][0].cpu().numpy()
                amplitudes = np.abs(act)
                return {
                    'amplitudes': amplitudes,
                    'channel_norms': [np.linalg.norm(amp) for amp in amplitudes],
                    'channel_entropy': [self._entropy(amp.flatten()) for amp in amplitudes]
                }
        return None

    def _entropy(self, p):
        p = p / (p.sum() + 1e-8)
        return -np.sum(p * np.log(p + 1e-8))


# ─── Global Model Initialisation ───────────────────────────────────────────────
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
MODEL_PATH = 'workingBEST_QCNN_MODEL.pth'          # Update with actual path
CAP_VALUE = 2.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    print("Loading QCNN model...")
    model = QCNN(Cap=CAP_VALUE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✓ QCNN model loaded successfully")
except Exception as e:
    print(f"✗ Error loading QCNN model: {e}")
    model = None

# Image transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def denormalize(tensor):
    """Convert normalized tensor back to displayable image (HWC, 0-1)."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)


# ─── NEW: Severity & Prognosis Assessment ─────────────────────────────────────
def assess_severity_and_prognosis(predicted_class, confidence):
    """
    Rule‑based severity and prognosis.
    Modify thresholds and messages according to your clinical knowledge.
    """
    if predicted_class == 'NORMAL':
        severity = 'None'
        prognosis = 'No abnormality detected. Routine follow‑up not required.'
    elif predicted_class == 'COVID19':
        severity = 'Severe' if confidence > 0.8 else 'Moderate'
        prognosis = 'Potential COVID‑19 infection. Isolation and further testing advised.'
    elif predicted_class == 'PNEUMONIA':
        severity = 'Moderate' if confidence > 0.7 else 'Mild'
        prognosis = 'Bacterial/viral pneumonia suspected. Antibiotics/antivirals may be needed.'
    elif predicted_class == 'TURBERCULOSIS':
        severity = 'Moderate' if confidence > 0.7 else 'Mild'
        prognosis = 'Possible tuberculosis. Sputum test and longer‑term treatment required.'
    else:
        severity = 'Unknown'
        prognosis = 'Unable to determine severity – consult specialist.'

    return severity, prognosis


# ─── Prediction & XAI Function ─────────────────────────────────────────────────
def analyze_xray(image_path):
    """Run QCNN prediction, XAI visualisation, and severity/prognosis."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]

    # Top‑3
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(CLASS_NAMES[i], probs[i]) for i in top3_idx]

    # Severity & prognosis (NEW)
    severity, prognosis = assess_severity_and_prognosis(CLASS_NAMES[pred_class], confidence)

    # XAI analysis
    analyzer = XQAI_Analyzer(model, device=device)

    saliency = analyzer.quantum_saliency_map(input_tensor, target_class=pred_class)
    qcam = analyzer.quantum_cam(input_tensor, target_class=pred_class)
    relevance_maps = analyzer.lrp_analysis(input_tensor, target_class=pred_class)
    quantum_state = analyzer.visualize_quantum_states(input_tensor)

    # Create a composite figure
    img_display = denormalize(input_tensor)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    axes[0].imshow(img_display)
    axes[0].set_title(f"Input\n{CLASS_NAMES[pred_class]} ({confidence:.2%})")
    axes[0].axis('off')

    axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title("Saliency Map")
    axes[1].axis('off')

    axes[2].imshow(qcam, cmap='viridis')
    axes[2].set_title("Q‑CAM")
    axes[2].axis('off')

    if 'input' in relevance_maps:
        inp_rel = relevance_maps['input']
        if inp_rel.ndim == 3:
            inp_rel = np.mean(np.abs(inp_rel), axis=0)
        axes[3].imshow(inp_rel, cmap='RdBu_r')
        axes[3].set_title("Input Relevance (LRP)")
        axes[3].axis('off')
    else:
        axes[3].text(0.5, 0.5, "N/A", ha='center')

    if 'conv1' in relevance_maps:
        c1 = np.mean(np.abs(relevance_maps['conv1']), axis=0)
        axes[4].imshow(c1, cmap='RdBu_r')
        axes[4].set_title("Conv1 Relevance")
        axes[4].axis('off')
    else:
        axes[4].text(0.5, 0.5, "N/A", ha='center')

    if 'conv2' in relevance_maps:
        c2 = np.mean(np.abs(relevance_maps['conv2']), axis=0)
        axes[5].imshow(c2, cmap='RdBu_r')
        axes[5].set_title("Conv2 Relevance")
        axes[5].axis('off')
    else:
        axes[5].text(0.5, 0.5, "N/A", ha='center')

    if quantum_state:
        axes[6].hist(quantum_state['amplitudes'].flatten(), bins=50, color='blue', alpha=0.7)
        axes[6].set_xlabel("Amplitude")
        axes[6].set_ylabel("Frequency")
        axes[6].set_title("Amplitude Distribution")
    else:
        axes[6].text(0.5, 0.5, "No data", ha='center')

    if quantum_state:
        norms = quantum_state['channel_norms']
        axes[7].bar(range(len(norms)), norms, color='green')
        axes[7].set_xlabel("Channel")
        axes[7].set_ylabel("L2 Norm")
        axes[7].set_title("Channel Norms")
    else:
        axes[7].text(0.5, 0.5, "No data", ha='center')

    plt.tight_layout()

    # Save the composite image
    xai_path = os.path.join('static/img', 'xai_' + os.path.basename(image_path))
    plt.savefig(xai_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'pred_class': CLASS_NAMES[pred_class],
        'confidence': float(confidence),
        'top3': top3,
        'xai_path': xai_path,
        'severity': severity,       # NEW
        'prognosis': prognosis       # NEW
    }


# ─── Upload & Predict Route ───────────────────────────────────────────────────
@app.route('/upload', methods=["GET", "POST"])
def upload():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("upload.html", message="No file uploaded.")

        myfile = request.files['file']
        fn = myfile.filename

        if fn == '':
            return render_template("upload.html", message="No file selected.")

        accepted = {'jpg', 'jpeg', 'png', 'jfif'}
        if fn.rsplit('.', 1)[-1].lower() not in accepted:
            return render_template("upload.html", message="Only image files accepted.")

        # Save uploaded image
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_fn = secure_filename(fn)
        unique_fn = f"{timestamp}_{safe_fn}"
        image_path = os.path.join('static/img', unique_fn)
        myfile.save(image_path)

        try:
            # Run analysis
            result = analyze_xray(image_path)

            # Prepare prediction dict for template (including new fields)
            prediction = {
                'class': result['pred_class'],
                'confidence': f"{result['confidence']:.2%}",
                'top3': [(c, f"{p:.2%}") for c, p in result['top3']],
                'severity': result['severity'],
                'prognosis': result['prognosis']
            }

            # Save to history (with new columns)
            if 'user_email' in session:
                try:
                    insert = """
                        INSERT INTO analysis_history
                        (user_email, filename, image_path, xai_path, predicted_class, confidence,
                         top1_class, top2_class, top3_class, top1_conf, top2_conf, top3_conf,
                         severity, prognosis)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    top3 = result['top3']  # list of (class, prob)
                    values = (
                        session['user_email'],
                        fn,
                        image_path,
                        result['xai_path'],
                        result['pred_class'],
                        f"{result['confidence']:.2%}",
                        top3[0][0] if len(top3) > 0 else None,
                        top3[1][0] if len(top3) > 1 else None,
                        top3[2][0] if len(top3) > 2 else None,
                        float(top3[0][1]) if len(top3) > 0 else 0.0,
                        float(top3[1][1]) if len(top3) > 1 else 0.0,
                        float(top3[2][1]) if len(top3) > 2 else 0.0,
                        result['severity'],
                        result['prognosis']
                    )
                    executionquery(insert, values)
                except Exception as e:
                    print(f"History save error: {e}")

            return render_template('upload.html',
                                   prediction=prediction,
                                   image_path=image_path,
                                   xai_path=result['xai_path'])

        except Exception as e:
            return render_template("upload.html", message=f"Error processing image: {str(e)}")

    return render_template('upload.html')


# ─── History Route ─────────────────────────────────────────────────────────────
@app.route('/history')
def history():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    user_email = session['user_email']

    # Fetch all history including new columns
    query = """
        SELECT id, filename, image_path, xai_path, predicted_class, confidence,
               top1_class, top2_class, top3_class, top1_conf, top2_conf, top3_conf,
               severity, prognosis, date
        FROM analysis_history
        WHERE user_email = %s
        ORDER BY date DESC
    """
    history_data = retrivequery1(query, (user_email,))

    # Statistics
    total = len(history_data)
    current_month = datetime.datetime.now().strftime("%Y-%m")
    this_month = sum(1 for row in history_data if row[14].strftime("%Y-%m") == current_month)  # date index 14

    stats = {'total': total, 'this_month': this_month}

    # Format for template
    formatted = []
    for row in history_data:
        formatted.append({
            'id': row[0],
            'filename': row[1],
            'image_path': row[2],
            'xai_path': row[3],
            'predicted_class': row[4],
            'confidence': row[5],
            'top1': (row[6], f"{row[9]:.2%}") if row[6] else None,
            'top2': (row[7], f"{row[10]:.2%}") if row[7] else None,
            'top3': (row[8], f"{row[11]:.2%}") if row[8] else None,
            'severity': row[12],
            'prognosis': row[13],
            'date': row[14].strftime("%Y-%m-%d %H:%M")
        })

    return render_template('history.html', history=formatted, stats=stats)


# ─── View Single Analysis ─────────────────────────────────────────────────────
@app.route('/view_analysis/<int:analysis_id>')
def view_analysis(analysis_id):
    if 'user_email' not in session:
        return redirect(url_for('login'))

    user_email = session['user_email']
    query = """
        SELECT filename, image_path, xai_path, predicted_class, confidence,
               top1_class, top2_class, top3_class, top1_conf, top2_conf, top3_conf,
               severity, prognosis
        FROM analysis_history
        WHERE id = %s AND user_email = %s
    """
    result = retrivequery1(query, (analysis_id, user_email))
    if not result:
        return redirect(url_for('history'))

    row = result[0]
    top3 = []
    if row[5]:
        top3.append((row[5], f"{row[8]:.2%}"))
    if row[6]:
        top3.append((row[6], f"{row[9]:.2%}"))
    if row[7]:
        top3.append((row[7], f"{row[10]:.2%}"))

    prediction = {
        'class': row[3],
        'confidence': row[4],
        'top3': top3,
        'severity': row[11],
        'prognosis': row[12]
    }

    return render_template('upload.html',
                           prediction=prediction,
                           image_path=row[1],
                           xai_path=row[2])


# ─── Delete Analysis ──────────────────────────────────────────────────────────
@app.route('/delete_analysis/<int:analysis_id>', methods=['POST'])
def delete_analysis(analysis_id):
    if 'user_email' not in session:
        return {'success': False, 'message': 'Not logged in'}, 401

    user_email = session['user_email']
    try:
        executionquery("DELETE FROM analysis_history WHERE id = %s AND user_email = %s",
                       (analysis_id, user_email))
        return {'success': True, 'message': 'Deleted'}
    except Exception as e:
        return {'success': False, 'message': str(e)}, 500


# ─── Clear All History ────────────────────────────────────────────────────────
@app.route('/clear_history', methods=['POST'])
def clear_history():
    if 'user_email' not in session:
        return {'success': False, 'message': 'Not logged in'}, 401

    user_email = session['user_email']
    try:
        executionquery("DELETE FROM analysis_history WHERE user_email = %s", (user_email,))
        return {'success': True, 'message': 'All history cleared'}
    except Exception as e:
        return {'success': False, 'message': str(e)}, 500


# ─── Download Report (updated with severity & prognosis) ──────────────────────
@app.route('/download_report/<int:analysis_id>')
def download_report(analysis_id):
    if 'user_email' not in session:
        return redirect(url_for('login'))

    user_email = session['user_email']
    query = """
        SELECT filename, predicted_class, confidence, top1_class, top2_class, top3_class,
               top1_conf, top2_conf, top3_conf, date, image_path, severity, prognosis
        FROM analysis_history
        WHERE id = %s AND user_email = %s
    """
    result = retrivequery1(query, (analysis_id, user_email))
    if not result:
        return redirect(url_for('history'))

    row = result[0]
    top1 = f"{row[3]} ({row[6]:.2%})" if row[3] else "N/A"
    top2 = f"{row[4]} ({row[7]:.2%})" if row[4] else "N/A"
    top3 = f"{row[5]} ({row[8]:.2%})" if row[5] else "N/A"

    report_html = f"""<!DOCTYPE html>
<html>
<head><title>X‑Ray Analysis Report #{analysis_id}</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    .header {{ background: linear-gradient(135deg, #0891b2, #066a80); color: white; padding: 20px; border-radius: 10px; }}
    .info-box {{ background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0; }}
    .label {{ font-weight: bold; color: #0891b2; }}
    .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
</style>
</head>
<body>
<div class="header"><h1>🫁 X‑Ray Analysis Report</h1><p>Report ID: #{analysis_id} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p></div>
<div class="info-box"><p><span class="label">Filename:</span> {row[0]}</p>
<p><span class="label">Analysis Date:</span> {row[9]}</p></div>
<div class="info-box"><h3>Results</h3>
<p><span class="label">Predicted Class:</span> {row[1]}</p>
<p><span class="label">Confidence:</span> {row[2]}</p>
<p><span class="label">Severity:</span> {row[11]}</p>
<p><span class="label">Prognosis:</span> {row[12]}</p>
<p><span class="label">Top‑3:</span><br>1. {top1}<br>2. {top2}<br>3. {top3}</p></div>
<div class="info-box"><p><span class="label">Model:</span> Explainable Quantum CNN (QCNN)</p>
<p><span class="label">XAI:</span> Saliency, Q‑CAM, LRP, Quantum state analysis</p></div>
<div class="footer"><p>Generated by AerialSense – chest X‑ray analysis with severity & prognosis</p></div>
</body>
</html>"""

    return Response(report_html, mimetype="text/html",
                    headers={"Content-disposition": f"attachment; filename=xray_report_{analysis_id}.html"})


if __name__ == '__main__':
    os.makedirs('static/img', exist_ok=True)
    app.run(debug=True)