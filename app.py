"""
Brain Tumor Detection Flask Backend with Grad-CAM XAI, Admin Functions, and Comprehensive Medical Data
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from io import BytesIO
import base64
import json
from datetime import datetime
from config import config, CLASS_DESCRIPTIONS
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# --- ADDITIONAL IMPORTS FOR EMAIL OTP ---
import random
import smtplib
import ssl
from email.message import EmailMessage

# ==================== COMPREHENSIVE MEDICAL DATA (Matched to Screenshots) ====================
# We update the imported CLASS_DESCRIPTIONS with the specific data from the images
CLASS_DESCRIPTIONS = {
    "meningioma_tumor": {
        "name": "Meningioma Tumor",
        "severity": "Medium",
        "detailed_description": "Meningiomas are tumors arising from the dura mater and arachnoid membrane that surround and protect the brain and spinal cord. Most are benign but can be serious due to compression of brain tissue. Characteristics: Typically slow-growing - Usually benign (80-90%) - Can compress brain tissue - May affect cerebrospinal fluid circulation",
        "symptoms": [
            "Headaches",
            "Vision problems (double or blurred)",
            "Hearing loss (ringing in ears)",
            "Weakness in legs or arms",
            "Seizures",
            "Balance and coordination problems",
            "Personality or behavioral changes"
        ],
        "causes": [
            "Unknown primary cause",
            "Possible radiation exposure history",
            "Hormonal factors (more common in women)",
            "Genetic syndromes (neurofibromatosis type 2)"
        ],
        "effects_on_health": [
            "Brain compression and swelling",
            "Impaired cognitive function",
            "Motor and sensory deficits",
            "Hydrocephalus (fluid accumulation)",
            "Quality of life impacts",
            "Generally better outcomes than gliomas"
        ],
        "treatment_options": [
            "Surgical resection (primary treatment)",
            "Radiation therapy if surgery incomplete",
            "Observation (for slow-growing benign tumors)",
            "Chemotherapy (rare)",
            "Stereotactic radiosurgery"
        ],
        "prognosis": "Generally favorable for benign meningiomas. 5-year survival rate: 85-90%. Malignant meningiomas have poorer outcomes.",
        "urgency": "URGENCY LEVEL: URGENT - Requires neurosurgical consultation and treatment planning"
    },
    "glioma_tumor": {
        "name": "Glioma Tumor",
        "severity": "High",
        "detailed_description": "Gliomas are tumors that start in the glial cells of the brain or spine. They are often aggressive and can infiltrate surrounding healthy brain tissue rapidly.",
        "symptoms": ["Progressive headaches", "Nausea and vomiting", "Confusion", "Memory loss", "Seizures"],
        "causes": ["Genetic mutations", "Environmental factors", "Radiation exposure"],
        "effects_on_health": ["High intracranial pressure", "Permanent neurological damage", "Cognitive decline"],
        "treatment_options": ["Surgery", "Radiation", "Chemotherapy", "Targeted therapy"],
        "prognosis": "Prognosis varies by grade. High-grade gliomas require aggressive intervention.",
        "urgency": "URGENCY LEVEL: CRITICAL - Immediate neurological intervention required"
    },
    "pituitary_tumor": {
        "name": "Pituitary Tumor",
        "severity": "Medium",
        "detailed_description": "Pituitary tumors are abnormal growths that develop in your pituitary gland. They can cause hormonal imbalances and vision loss by pressing on optic nerves.",
        "symptoms": ["Peripheral vision loss", "Hormonal disorders", "Fatigue", "Weight changes"],
        "causes": ["Genetic predisposition", "MEN1 syndrome", "Spontaneous mutations"],
        "effects_on_health": ["Hormonal disorders", "Vision impairment", "Pituitary apoplexy"],
        "treatment_options": ["Endoscopic surgery", "Hormone therapy", "Radiation"],
        "prognosis": "Highly treatable. Most patients return to normal life with hormone management.",
        "urgency": "URGENCY LEVEL: ELEVATED - Requires endocrinology and neurosurgical review"
    },
    "no_tumor": {
        "name": "No Tumor Detected",
        "severity": "Low",
        "detailed_description": "The AI analysis indicates a healthy brain scan with no significant masses, lesions, or tumors detected within the provided image parameters.",
        "symptoms": ["N/A"],
        "causes": ["N/A"],
        "effects_on_health": ["Healthy physiological brain function"],
        "treatment_options": ["Routine check-ups as recommended by physician"],
        "prognosis": "Excellent clinical outlook.",
        "urgency": "URGENCY LEVEL: ROUTINE - No immediate oncology intervention required"
    }
}

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "neurovision_secret_key" # Added for session management
CORS(app)

DB_NAME = "users.db"

# ==================== EMAIL CONFIGURATION ====================
# Update these with your real credentials to send real emails
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = "alerts.braintumor.ai@gmail.com"
SENDER_PASSWORD = "edso xbmy pkmk goqr"
# Provide your App Password here

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    return conn

def init_db():
    """Initializes the database and creates the users table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            about_me TEXT DEFAULT 'Add some information about yourself',
            profile_pic TEXT,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Check if columns exist (for migration if table already exists)
    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'about_me' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN about_me TEXT DEFAULT 'Add some information about yourself'")
    if 'profile_pic' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN profile_pic TEXT")
    if 'is_admin' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")

    # Table for temporary registration data during OTP phase
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pending_users (
            email TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            otp TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Scans Table for Dashboard
    cursor.execute('''
           CREATE TABLE IF NOT EXISTS scans (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               user_email TEXT NOT NULL,
               filename TEXT NOT NULL,
               predicted_class TEXT NOT NULL,
               confidence REAL NOT NULL,
               timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )
       ''')

    # Check if admin user exists, if not create a default one
    cursor.execute("SELECT * FROM users WHERE email = 'admin@neurovision.ai'")
    if not cursor.fetchone():
        hashed_pw = generate_password_hash("admin123")
        cursor.execute("INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)",
                       ("System Admin", "admin@neurovision.ai", hashed_pw, 1))

    conn.commit()
    conn.close()

def add_user(username, email, password):
    """Hashes the password and stores a new user in the database."""
    hashed_password = generate_password_hash(password)
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed_password)
        )
        conn.commit()
        return True, "User registered successfully"
    except sqlite3.IntegrityError:
        return False, "Email already exists"
    finally:
        conn.close()

def verify_user(email, password):
    """Checks if the user exists and the password matches the hash."""
    conn = get_db_connection()
    cursor = conn.cursor()
    user = cursor.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if user and check_password_hash(user['password'], password):
        return user
    return None

# ==================== OTP HELPER FUNCTIONS ====================

def send_otp_email(target_email, otp_code):
    """Sends a real email containing the 6-digit verification code"""
    msg = EmailMessage()
    msg['Subject'] = "Verification Code - Brain Tumor Prediction"
    msg['From'] = SENDER_EMAIL
    msg['To'] = target_email
    msg.set_content(f"Hello,\n\nYour 6-digit verification code is: {otp_code}\n\nIf you did not request this, please ignore this email.")

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"SMTP Error: {e}")
        return False

# Load configuration
app.config.update(
    UPLOAD_FOLDER=config.UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH=config.MAX_CONTENT_LENGTH,
)

# Ensure upload folder exists
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# Global variables for model
model = None
class_names = None
last_conv_layer = config.LAST_CONV_LAYER


# ==================== UTILITY FUNCTIONS ====================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


def resize_with_padding(image, target_size=config.IMG_SIZE):
    """Resize image with padding to maintain aspect ratio"""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def preprocess_image(image):
    """Preprocess image: CLAHE enhancement"""
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # FIXED: Corrected typo (v2 -> cv2) to prevent prediction crash seen in terminal logs
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced.astype(np.float32)


def generate_gradcam(img_array, model, last_conv_layer_name):
    """Generate Grad-CAM heatmap for explainability"""
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        # Calculate loss for the predicted class
        # Use the maximum prediction across the batch
        loss = tf.reduce_max(preds)

    grads = tape.gradient(loss, conv_out)

    if grads is None:
        return np.zeros((300, 300))

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    if tf.reduce_max(heatmap) > 0:
        heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()


def heatmap_to_base64(heatmap, original_img):
    """Convert heatmap and overlay to base64 string"""
    hm_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    jet_heatmap = (plt.cm.jet(hm_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(original_img, 0.6, jet_heatmap, 0.4, 0)

    # Convert to base64
    _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return f"data:image/png;base64,{img_base64}"


def generate_pdf_report(prediction_data, gradcam_base64, tumor_info):
    """Generate a comprehensive PDF report with prediction results and medical information"""
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    story = []

    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2563eb'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=10,
        leading=14,
        alignment=TA_JUSTIFY
    )

    # Title
    story.append(Paragraph("Brain Tumor Detection Report", title_style))
    story.append(Spacer(1, 0.3 * inch))

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Report Generated:</b> {timestamp}", normal_style))
    story.append(Spacer(1, 0.2 * inch))

    # === PREDICTION SECTION ===
    story.append(Paragraph("1. PREDICTION RESULTS", heading_style))
    story.append(Spacer(1, 0.1 * inch))

    # Prediction table
    pred_data = [
        ['Metric', 'Value'],
        ['Predicted Class', prediction_data['predicted_class'].replace('_', ' ').title()],
        ['Confidence Score', f"{prediction_data['confidence']}%"],
        ['Model', 'EfficientNetB3'],
        ['Explainability Method', 'Grad-CAM (Gradient-weighted Class Activation Mapping)']
    ]

    pred_table = Table(pred_data, colWidths=[2.5 * inch, 3.5 * inch])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 0.2 * inch))

    # All confidence scores
    story.append(Paragraph("<b>Detailed Confidence Scores:</b>", normal_style))
    scores_data = [['Class', 'Confidence']]
    for class_name, score in prediction_data['all_predictions'].items():
        scores_data.append([class_name.replace('_', ' ').title(), f"{score:.2f}%"])

    scores_table = Table(scores_data, colWidths=[3 * inch, 3 * inch])
    scores_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0fdf4')])
    ]))
    story.append(scores_table)
    story.append(Spacer(1, 0.3 * inch))

    # === GRAD-CAM VISUALIZATION ===
    story.append(Paragraph("2. GRAD-CAM VISUALIZATION", heading_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "The heatmap below shows which regions of the MRI image contributed most to the model's prediction. "
        "Warmer colors (red/yellow) indicate areas the model focused on.",
        normal_style
    ))
    story.append(Spacer(1, 0.15 * inch))

    # Add Grad-CAM image
    try:
        # Convert base64 to image
        base64_data = gradcam_base64.replace('data:image/png;base64,', '')
        img_data = base64.b64decode(base64_data)
        img_buffer = BytesIO(img_data)
        img = Image(img_buffer, width=5 * inch, height=4 * inch)
        story.append(img)
    except:
        story.append(Paragraph("<i>Grad-CAM visualization could not be displayed</i>", normal_style))

    story.append(Spacer(1, 0.3 * inch))

    # === TUMOR INFORMATION ===
    if tumor_info:
        story.append(PageBreak())
        story.append(Paragraph("3. TUMOR INFORMATION & HEALTH EFFECTS", heading_style))
        story.append(Spacer(1, 0.1 * inch))

        # Tumor name and severity
        story.append(Paragraph(f"<b>Tumor Type:</b> {tumor_info.get('name', 'N/A')}", normal_style))
        story.append(Paragraph(f"<b>Severity Level:</b> {tumor_info.get('severity', 'N/A')}", normal_style))
        story.append(
            Paragraph(f"<b>Urgency:</b> <font color=red>{tumor_info.get('urgency', 'N/A')}</font>", normal_style))
        story.append(Spacer(1, 0.15 * inch))

        # Description
        story.append(Paragraph("<b>Medical Description:</b>", normal_style))
        story.append(Paragraph(tumor_info.get('detailed_description', ''), normal_style))
        story.append(Spacer(1, 0.15 * inch))

        # Symptoms
        if tumor_info.get('symptoms'):
            story.append(Paragraph("<b>Common Symptoms:</b>", normal_style))
            for symptom in tumor_info.get('symptoms', []):
                story.append(Paragraph(f"• {symptom}", normal_style))
            story.append(Spacer(1, 0.1 * inch))

        # Causes
        if tumor_info.get('causes'):
            story.append(Paragraph("<b>Root Causes & Risk Factors:</b>", normal_style))
            for cause in tumor_info.get('causes', []):
                story.append(Paragraph(f"• {cause}", normal_style))
            story.append(Spacer(1, 0.1 * inch))

        # Health effects
        if tumor_info.get('effects_on_health'):
            story.append(Paragraph("<b>Effects on Human Health:</b>", normal_style))
            for effect in tumor_info.get('effects_on_health', []):
                story.append(Paragraph(f"• {effect}", normal_style))
            story.append(Spacer(1, 0.1 * inch))

        # Treatment options
        if tumor_info.get('treatment_options'):
            story.append(Paragraph("<b>Treatment Options:</b>", normal_style))
            for treatment in tumor_info.get('treatment_options', []):
                story.append(Paragraph(f"• {treatment}", normal_style))
            story.append(Spacer(1, 0.1 * inch))

        # Prognosis
        if tumor_info.get('prognosis'):
            story.append(Paragraph("<b>Prognosis:</b>", normal_style))
            story.append(Paragraph(tumor_info.get('prognosis', ''), normal_style))
            story.append(Spacer(1, 0.15 * inch))

    # === INSTRUCTIONS & DISCLAIMER ===
    story.append(PageBreak())
    story.append(Paragraph("4. IMPORTANT INSTRUCTIONS & DISCLAIMER", heading_style))
    story.append(Spacer(1, 0.1 * inch))

    instructions = [
        "<b>Medical Disclaimer:</b><br/>This AI-powered analysis is intended for informational purposes only and should NOT be used as a substitute for professional medical diagnosis or treatment. Always consult with a qualified healthcare professional before making any medical decisions.",
        "<br/><b>How to Use This Report:</b><br/>1. Review the prediction results and confidence scores<br/>2. Understand the model's focus areas using the Grad-CAM visualization<br/>3. Read the tumor information provided for medical context<br/>4. Discuss results with your healthcare provider<br/>5. Seek professional medical evaluation for confirmation",
        "<br/><b>Accuracy Notes:</b><br/>• The model was trained on EfficientNetB3 architecture<br/>• Accuracy depends on image quality and resolution<br/>• This tool should be used as a secondary screening aid<br/>• Professional radiologist review is essential for clinical decisions",
        "<br/><b>Important Reminders:</b><br/>• Early detection and treatment are crucial for tumor management<br/>• Seek immediate medical attention if experiencing severe symptoms<br/>• This report should be shared with healthcare professionals<br/>• Do not delay professional medical consultation based on this analysis"
    ]

    for instruction in instructions:
        story.append(Paragraph(instruction, normal_style))
        story.append(Spacer(1, 0.1 * inch))

    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer


# ==================== MODEL LOADING ====================

def load_model_and_classes():
    """Load the trained model and class names"""
    global model, class_names

    try:
        # Try loading the best model first
        model_path = config.MODEL_PATH
        if not os.path.exists(model_path):
            model_path = config.FALLBACK_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found. Looking for {model_path}")

        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully. Input shape: {model.input_shape}")

        # Use class names from config
        class_names = config.CLASS_NAMES
        print(f"✓ Class names loaded: {class_names}")

        return True
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False


# ==================== FLASK ROUTES ====================

# Authentication API Routes
@app.route('/api/send-otp', methods=['POST'])
def handle_send_otp():
    """Generates an OTP and sends it via email."""
    data = request.get_json()
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')

    if not email or not username or not password:
        return jsonify({'error': 'Missing required fields'}), 400

    otp = str(random.randint(100000, 999999))

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Save temporary user data
        cursor.execute(
            "INSERT OR REPLACE INTO pending_users (email, username, password, otp) VALUES (?, ?, ?, ?)",
            (email, username, password, otp)
        )
        conn.commit()

        if send_otp_email(email, otp):
            return jsonify({'message': 'OTP sent successfully'}), 200
        else:
            return jsonify({'error': 'Failed to send email. Check SMTP settings.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/verify-registration', methods=['POST'])
def verify_registration():
    """Verifies OTP and finalizes registration"""
    data = request.get_json()
    email = data.get('email')
    otp = data.get('otp')

    if not email or not otp:
        return jsonify({'error': 'Missing email or OTP'}), 400

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        pending = cursor.execute("SELECT * FROM pending_users WHERE email = ?", (email,)).fetchone()

        if pending and pending['otp'] == otp:
            # Move to main users table
            success, message = add_user(pending['username'], pending['email'], pending['password'])
            if success:
                cursor.execute("DELETE FROM pending_users WHERE email = ?", (email,))
                conn.commit()
                return jsonify({'message': 'Registration successful!'}), 201
            else:
                return jsonify({'error': message}), 409
        else:
            return jsonify({'error': 'Invalid verification code'}), 401
    finally:
        conn.close()

@app.route('/api/register', methods=['POST'])
def register():
    """Endpoint for user registration"""
    data = request.get_json()
    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Missing required fields'}), 400

    success, message = add_user(data['username'], data['email'], data['password'])
    if success:
        return jsonify({'message': message}), 201
    else:
        return jsonify({'error': message}), 409

@app.route('/api/login', methods=['POST'])
def login():
    """Endpoint for user login"""
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Missing email or password'}), 400

    user = verify_user(data['email'], data['password'])
    if user:
        # SET SESSION FOR DASHBOARD
        session['user_email'] = user['email']
        session['username'] = user['username']
        session['is_admin'] = user['is_admin']
        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email']
            }
        }), 200
    else:
        return jsonify({'error': 'Invalid email or password'}), 401


@app.route('/userlist')
def serve_userlist_page():
    """Serves the User List page (per your requirement) and passes data"""
    # ADMIN REDIRECTORY GUARD
    if 'user_email' not in session or not session.get('is_admin'):
        return redirect(url_for('serve_admin_login'))

    conn = get_db_connection()
    try:
        # FIXED: This now fetches real users from the DB to fix the "Not showing" issue
        users = [dict(row) for row in conn.execute("SELECT id, username, email, profile_pic FROM users").fetchall()]
        return render_template('userslist.html', users_list=users)
    except Exception as e:
        return f"Database Error: {e}", 500
    finally:
        conn.close()

@app.route('/')
def serve_home_page():
    """FIXED: Serves the Home Page directly when hitting localhost:5000/ as requested"""
    return render_template('home.html')

@app.route('/about')
def serve_about_page():
    return render_template('about.html')

@app.route('/aboutt')
def serve_aboutt_page():
    return render_template('aboutt.html')

@app.route('/abouttt')
def serve_abouttt_page():
    return render_template('abouttt.html')



@app.route('/admin_dashboard')
def serve_admindashboard_page():
    """Serves the Admin Dashboard with Guard (Redirectories) - Fixed Typo in function name"""
    if 'user_email' not in session or not session.get('is_admin'):
        return redirect(url_for('serve_admin_login'))
    return redirect(url_for('serve_admin_dashboard'))

@app.route('/dashboard')
def serve_dashboard_page():
    """
    Dashboard route that fetches statistics and activities from the SQLite database.
    Calculates counts based on the logged-in user's email.
    """
    if 'user_email' not in session:
        return redirect(url_for('serve_login'))

    email = session['user_email']
    username = session.get('username', 'User')

    conn = get_db_connection()

    # 1. Classification stats for donut chart
    # Mapping exact classes from config.CLASS_NAMES
    classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    stats = {}
    for cls in classes:
        row = conn.execute('SELECT COUNT(*) FROM scans WHERE user_email = ? AND predicted_class = ?',
                           (email, cls)).fetchone()
        stats[cls] = row[0] if row else 0

    # 2. Total scans
    total_row = conn.execute('SELECT COUNT(*) FROM scans WHERE user_email = ?', (email,)).fetchone()
    total = total_row[0] if total_row else 0

    # 3. Recent activities (Latest 5 scans)
    activities = conn.execute('''
        SELECT id, predicted_class, timestamp 
        FROM scans 
        WHERE user_email = ? 
        ORDER BY timestamp DESC LIMIT 5
    ''', (email,)).fetchall()

    conn.close()

    return render_template('dashboard.html',
                           user_email=email,
                           username=username,
                           stats=stats,
                           total=total,
                           activities=activities,
                           trend=[92, 94, 93, 95, 96, 97]) # Simulated accuracy trend


@app.route('/login')
def serve_login():
    """Serves the Login Page"""
    return render_template('auth.html')

@app.route('/preview')
def detection_page():
    """Serves the Diagnostic Dashboard (Preview Page)"""
    return render_template('index.html')


@app.route('/profile')
def serve_profile_page():
    """
    STRICT DATABASE ACCESS: Fetches user details from the database
    using the session email to ensure about_me and profile_pic persist.
    """
    # Ensure user is logged in
    if 'user_email' not in session:
        return redirect(url_for('serve_login'))

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (session['user_email'],)).fetchone()
    conn.close()

    # Passing 'user' to the template so 'user.about_me' and 'user.profile_pic' can be accessed
    return render_template('profile.html', user=user)

@app.route('/performance')
def performance():
    """Serves the Performance Analysis Page"""
    return render_template('performance.html')


# ==================== ADMIN SYSTEM LOGIC ====================
@app.route('/admin/list')
def serve_admin_list_view():
    # ... session guards ...
    return render_template('adminslist.html') # Must match the filename exactly

@app.route('/admin/login')
def serve_admin_login():
    """Serves the Admin Login Page"""
    return render_template('admin_login.html')

@app.route('/api/admin/login', methods=['POST'])
def api_admin_login():
    """Specific API for Admin Authentication (Resolves Endpoint not found error)"""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = verify_user(email, password)
    if user and user['is_admin']:
        session['user_email'] = user['email']
        session['username'] = user['username']
        session['is_admin'] = 1
        return jsonify({'message': 'Admin login successful'}), 200
    return jsonify({'error': 'Unauthorized admin access'}), 401

@app.route('/admin/dashboard')
def serve_admin_dashboard():
    """Admin Dashboard route fetching real-time stats from SQLite with Guards"""
    # Security Redirect Guard: Only Admins can enter
    if 'user_email' not in session or not session.get('is_admin'):
        return redirect(url_for('serve_admin_login'))

    conn = get_db_connection()
    try:
        # 1. Global statistics
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        active_users = conn.execute("SELECT COUNT(DISTINCT user_email) FROM scans").fetchone()[0]
        new_users_30 = conn.execute("SELECT COUNT(*) FROM users WHERE created_at >= date('now', '-30 days')").fetchone()[0]
        total_scans = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]

        # 2. Chart distribution data
        rows = conn.execute("SELECT predicted_class, COUNT(*) as count FROM scans GROUP BY predicted_class").fetchall()
        dist_stats = {'glioma': 0, 'meningioma': 0, 'no_tumor': 0, 'pituitary': 0}
        for row in rows:
            p_class = row['predicted_class']
            if p_class:
                cls = p_class.lower()
                if 'glioma' in cls: dist_stats['glioma'] = row['count']
                elif 'meningioma' in cls: dist_stats['meningioma'] = row['count']
                elif 'no' in cls: dist_stats['no_tumor'] = row['count']
                elif 'pituitary' in cls: dist_stats['pituitary'] = row['count']

        return render_template('admin_dashboard.html',
                               total_users=total_users,
                               active_users=active_users,
                               new_users_30=new_users_30,
                               total_scans=total_scans,
                               dist_stats=dist_stats)
    except Exception as e:
        print(f"Admin Dashboard Error: {e}")
        return "Internal Server Error", 500
    finally:
        conn.close()

@app.route('/admin/users')
def serve_admin_users_view():
    """Redirect to the user list with admin guard"""
    if 'user_email' not in session or not session.get('is_admin'):
        return redirect(url_for('serve_admin_login'))
    return redirect(url_for('serve_userlist_page'))

@app.route('/api/admin/stats')
def get_admin_stats_api():
    """API endpoint for dashboard updates (FIXED: Added this to solve 404)"""
    if 'user_email' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    conn = get_db_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        active = conn.execute("SELECT COUNT(DISTINCT user_email) FROM scans").fetchone()[0]
        total_scans = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]

        new_users_30 = conn.execute("SELECT COUNT(*) FROM users WHERE created_at >= date('now', '-30 days')").fetchone()[0]

        rows = conn.execute("SELECT predicted_class, COUNT(*) as count FROM scans GROUP BY predicted_class").fetchall()
        dist = {'glioma': 0, 'meningioma': 0, 'no_tumor': 0, 'pituitary': 0}
        for row in rows:
            p_class = row['predicted_class']
            if p_class:
                cls = p_class.lower()
                if 'glioma' in cls: dist['glioma'] = row['count']
                elif 'meningioma' in cls: dist['meningioma'] = row['count']
                elif 'no' in cls: dist['no_tumor'] = row['count']
                elif 'pituitary' in cls: dist['pituitary'] = row['count']

        return jsonify({
            'total_users': total,
            'active_users': active,
            'inactive_users': total - active,
            'new_users_30': new_users_30,
            'total_scans': total_scans,
            'dist': dist
        })
    finally:
        conn.close()

@app.route('/api/admin/users-list')
def api_users_list():
    """API endpoint to fetch users for real-time list updates (FIXED: Added this to solve 404)"""
    if 'user_email' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    conn = get_db_connection()
    try:
        users = [dict(row) for row in conn.execute("SELECT id, username, email, profile_pic FROM users").fetchall()]
        return jsonify(users)
    finally:
        conn.close()

# --- ADDED: ADMINS LIST ACCESS FUNCTIONALITY ---
@app.route('/api/admin/admins-list')
def api_admins_list():
    """API endpoint to fetch only clinical administrators from the database registry"""
    if 'user_email' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized admin access'}), 401
    conn = get_db_connection()
    try:
        # Accessing admin database details: WHERE is_admin = 1
        admins = [dict(row) for row in conn.execute(
            "SELECT id, username, email, about_me, created_at FROM users WHERE is_admin = 1"
        ).fetchall()]
        return jsonify(admins)
    finally:
        conn.close()

@app.route('/api/admin/user/<int:user_id>/delete', methods=['DELETE'])
def api_delete_user(user_id):
    """API endpoint to delete a user"""
    if 'user_email' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return jsonify({'status': 'deleted'}), 200
    finally:
        conn.close()

# --- ADDED: Endpoint to manually create a new admin from the dashboard ---
@app.route('/api/admin/create', methods=['POST'])
def api_admin_create():
    """Creates a new admin record in the database."""
    # Guard: only authorized admins can add more admins
    if 'user_email' not in session or not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized admin access'}), 401

    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'Standard Admin')

    if not all([name, email, password]):
        return jsonify({'error': 'Missing required fields'}), 400

    hashed_pw = generate_password_hash(password)
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Insert as admin (is_admin = 1)
        cursor.execute(
            "INSERT INTO users (username, email, password, is_admin, about_me) VALUES (?, ?, ?, ?, ?)",
            (name, email, hashed_pw, 1, f"Role: {role}")
        )
        conn.commit()
        return jsonify({'message': 'New admin created successfully'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already exists in database'}), 409
    finally:
        conn.close()

# ==================== PREDICTION API ====================

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict tumor class and generate Grad-CAM explanation.
    Now logs every prediction into the 'scans' table for dashboard statistics.
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {", ".join(config.ALLOWED_EXTENSIONS)}'}), 400

        # Read image from file
        in_memory_file = file.stream.read()
        data = np.frombuffer(in_memory_file, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Could not read image file'}), 400

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img_rgb.shape

        # Preprocess
        img_resized = resize_with_padding(img_rgb, config.IMG_SIZE)
        processed = preprocess_image(img_resized)
        img_batch = np.expand_dims(processed, axis=0)

        # Make prediction
        predictions = model.predict(img_batch, verbose=0)

        # Handle different tensor formats
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0] if len(predictions) > 0 else predictions

        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()

        predictions = np.array(predictions)
        if predictions.ndim > 1:
            predictions = predictions[0]

        predicted_idx = int(np.argmax(predictions))
        predicted_class = class_names[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100
        confidence_rounded = round(confidence, 2)

        # LOG TO DATABASE IF USER IS LOGGED IN
        if 'user_email' in session:
            conn = get_db_connection()
            conn.execute('''
                INSERT INTO scans (user_email, filename, predicted_class, confidence)
                VALUES (?, ?, ?, ?)
            ''', (session['user_email'], secure_filename(file.filename), predicted_class, confidence_rounded))
            conn.commit()
            conn.close()

        # Generate Grad-CAM
        heatmap = generate_gradcam(img_batch, model, last_conv_layer)
        gradcam_img = heatmap_to_base64(heatmap, img_resized)

        # Get confidence scores for all classes
        confidence_scores = {
            class_names[i]: float(predictions[i]) * 100
            for i in range(len(class_names))
        }

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
        cv2.imwrite(filepath, img)

        return jsonify({
            'status': 'success',
            'predicted_class': predicted_class,
            'confidence': confidence_rounded,
            'confidence_scores': {k: round(v, 2) for k, v in confidence_scores.items()},
            'gradcam': gradcam_img,
            'all_predictions': confidence_scores,
            'input_shape': list(original_shape),
            'timestamp': datetime.now().isoformat(),
            'uploaded_file': filepath
        }), 200

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/generate-pdf-report', methods=['POST'])
def generate_pdf_report_endpoint():
    """
    Generate a comprehensive PDF report with prediction results and medical information
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        predicted_class = data.get('predicted_class')
        confidence = data.get('confidence')
        all_predictions = data.get('all_predictions', {})
        gradcam_base64 = data.get('gradcam', '')

        if not predicted_class:
            return jsonify({'error': 'Missing predicted class'}), 400

        # Get tumor information
        tumor_info = CLASS_DESCRIPTIONS.get(predicted_class)

        # Prepare prediction data for PDF
        prediction_data = {
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2) if confidence else 0,
            'all_predictions': all_predictions
        }

        # Generate PDF
        pdf_buffer = generate_pdf_report(prediction_data, gradcam_base64, tumor_info)

        # Return as file download
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'brain_tumor_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )

    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple files
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503

        if 'files' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        files = request.files.getlist('files')
        results = []

        for file in files:
            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'message': 'File type not allowed'
                })
                continue

            try:
                # Read and process image
                in_memory_file = file.stream.read()
                data = np.frombuffer(in_memory_file, np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)

                if img is None:
                    results.append({
                        'filename': file.filename,
                        'status': 'error',
                        'message': 'Could not read image'
                    })
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = resize_with_padding(img_rgb, config.IMG_SIZE)
                processed = preprocess_image(img_resized)
                img_batch = np.expand_dims(processed, axis=0)

                # Predict
                predictions = model.predict(img_batch, verbose=0)
                predicted_idx = np.argmax(predictions[0])

                results.append({
                    'filename': file.filename,
                    'status': 'success',
                    'predicted_class': class_names[predicted_idx],
                    'confidence': round(float(predictions[0][predicted_idx]) * 100, 2)
                })

            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'message': str(e)
                })

        return jsonify({
            'status': 'success',
            'total_files': len(files),
            'successful': len([r for r in results if r['status'] == 'success']),
            'results': results
        }), 200

    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/update-profile', methods=['POST'])
def update_profile():
    """Updates display name and about me text"""
    if 'user_email' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    new_username = data.get('username')
    new_about = data.get('about')
    email = session['user_email']

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET username = ?, about_me = ? WHERE email = ?",
                       (new_username, new_about, email))
        conn.commit()
        session['username'] = new_username
        return jsonify({'message': 'Profile updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/change-password', methods=['POST'])
def change_password():
    """Updates user password in database"""
    if 'user_email' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    new_password = data.get('new_password')

    if not new_password:
        return jsonify({'error': 'Password is required'}), 400

    hashed_password = generate_password_hash(new_password)
    email = session['user_email']

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_password, email))
        conn.commit()
        return jsonify({'message': 'Password updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/upload-profile-pic', methods=['POST'])
def upload_profile_pic():
    """Saves profile picture as base64 string to database"""
    if 'user_email' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    img_base64 = data.get('image')
    email = session['user_email']

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET profile_pic = ? WHERE email = ?", (img_base64, email))
        conn.commit()
        return jsonify({'message': 'Profile picture updated'}), 200
    finally:
        conn.close()

@app.route('/logout')
def handle_logout():
    """Clears session and redirects to login"""
    session.clear()
    return redirect(url_for('serve_login'))


# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 60)
    print("Brain Tumor Detection with Grad-CAM XAI")
    print("=" * 60)

    # Initialize SQLite Database
    init_db()
    print("✓ SQLite Database initialized.")

    # Load model
    if load_model_and_classes():
        print("\n✓ Starting Flask server...")
        print("=" * 60 + "\n")

        # Render requires dynamic port
        port = int(os.environ.get("PORT", 5000))

        app.run(host="0.0.0.0", port=port)

    else:
        print("\n✗ Failed to load model. Please check model files.")
        print("Expected: efficientnetb3_best.keras or efficientnetb3_final.keras")