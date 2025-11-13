from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import joblib
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from datetime import datetime
import pandas as pd
import io
import os

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')

# Configure database (SQLite file)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///predictions.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure Flask-Mail
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')
mail = Mail(app)

# Token serializer for password reset
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize database on startup
@app.before_request
def init_db():
    if not hasattr(app, 'db_initialized'):
        with app.app_context():
            db.create_all()
            
            # Create default admin if doesn't exist
            if not User.query.filter_by(username='admin').first():
                admin = User(username='admin', email='admin@example.com', role='admin', name='Administrator')
                admin.set_password('admin123')
                db.session.add(admin)
                db.session.commit()
            
            # Create default subjects if none exist
            if Subject.query.count() == 0:
                default_subjects = [
                    Subject(name='Python Programming', code='CS101', description='Introduction to Python'),
                    Subject(name='Database Systems', code='CS201', description='SQL and Database Design'),
                    Subject(name='Web Development', code='CS202', description='HTML, CSS, JavaScript'),
                    Subject(name='Data Structures', code='CS301', description='Algorithms and Data Structures'),
                ]
                for subject in default_subjects:
                    db.session.add(subject)
                db.session.commit()
            
            app.db_initialized = True

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ============ DATABASE MODELS ============

# User model (Admin, Lecturer, Student)
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'admin', 'lecturer', 'student'
    student_id = db.Column(db.String(50), nullable=True)  # Only for students
    name = db.Column(db.String(100), nullable=True)
    course = db.Column(db.String(100), nullable=True)
    year_of_study = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Subject/Topic model
class Subject(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    code = db.Column(db.String(20), nullable=False)
    description = db.Column(db.Text, nullable=True)
    
# Progress Entry model (multiple entries per student over time)
class ProgressEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    week_number = db.Column(db.Integer, nullable=False)  # Week 1, 2, 3, etc.
    semester = db.Column(db.String(20), nullable=False)  # e.g., "2025-1"
    attendance_percentage = db.Column(db.Float, nullable=False)
    participation_score = db.Column(db.Float, nullable=False)
    overall_score = db.Column(db.Float, nullable=True)  # Calculated
    risk_level = db.Column(db.String(50), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    student = db.relationship('User', backref='progress_entries')

# Topic Performance (performance per subject in each progress entry)
class TopicPerformance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    progress_entry_id = db.Column(db.Integer, db.ForeignKey('progress_entry.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=False)
    score = db.Column(db.Float, nullable=False)  # 0-100
    strengths = db.Column(db.Text, nullable=True)  # What they're good at
    weaknesses = db.Column(db.Text, nullable=True)  # What needs improvement
    
    progress_entry = db.relationship('ProgressEntry', backref='topic_performances')
    subject = db.relationship('Subject')

# Recommendations (AI-generated smart recommendations)
class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    progress_entry_id = db.Column(db.Integer, db.ForeignKey('progress_entry.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=True)
    recommendation_text = db.Column(db.Text, nullable=False)
    priority = db.Column(db.String(20), nullable=False)  # 'high', 'medium', 'low'
    
    progress_entry = db.relationship('ProgressEntry', backref='recommendations')
    subject = db.relationship('Subject')

# ============ HELPER FUNCTIONS ============

def calculate_risk_level(attendance, avg_topic_score, participation):
    """
    Calculate risk level WITHOUT exam scores
    Based on: Attendance (30%), Topic Performance (50%), Participation (20%)
    """
    overall_score = (attendance * 0.30) + (avg_topic_score * 0.50) + (participation * 0.20)
    
    if overall_score >= 85:
        return "Performing Well", overall_score, "success"
    elif overall_score >= 70:
        return "Low Risk", overall_score, "info"
    elif overall_score >= 50:
        return "Medium Risk", overall_score, "warning"
    else:
        return "High Risk", overall_score, "danger"

def generate_smart_recommendations(student, topic_performances, attendance, participation):
    """
    Generate SMART, specific recommendations based on topic performance
    """
    recommendations = []
    
    # Analyze each subject
    for tp in topic_performances:
        subject_name = tp.subject.name
        score = tp.score
        
        if score < 50:
            # Critical - needs urgent attention
            recommendations.append({
                'subject': subject_name,
                'priority': 'high',
                'text': f"🚨 URGENT: You're struggling with {subject_name} (Score: {score}%). Schedule extra tutoring sessions immediately. Focus on foundational concepts before moving to advanced topics."
            })
            
            # Specific material recommendations
            if 'Python' in subject_name or 'Programming' in subject_name:
                recommendations.append({
                    'subject': subject_name,
                    'priority': 'high',
                    'text': f"📚 For {subject_name}: Review Python basics on w3schools.com, complete exercises on HackerRank (Easy level), watch CS Dojo tutorials on YouTube."
                })
            elif 'Database' in subject_name:
                recommendations.append({
                    'subject': subject_name,
                    'priority': 'high',
                    'text': f"📚 For {subject_name}: Practice SQL queries on SQLZoo, review ER diagram examples, complete Khan Academy database course."
                })
            elif 'Web' in subject_name:
                recommendations.append({
                    'subject': subject_name,
                    'priority': 'high',
                    'text': f"📚 For {subject_name}: Build 3 simple HTML pages, practice CSS layouts on freeCodeCamp, complete JavaScript basics on Codecademy."
                })
        
        elif score < 70:
            # Moderate - needs improvement
            recommendations.append({
                'subject': subject_name,
                'priority': 'medium',
                'text': f"⚠️ {subject_name} needs improvement (Score: {score}%). Join study groups, complete all practice exercises, and ask questions during lectures."
            })
    
    # Attendance recommendations
    if attendance < 75:
        recommendations.append({
            'subject': 'General',
            'priority': 'high',
            'text': f"🚨 Attendance is critically low ({attendance}%). Missing classes means missing crucial explanations and examples. Aim for 85%+ attendance."
        })
    
    # Participation recommendations
    if participation < 50:
        recommendations.append({
            'subject': 'General',
            'priority': 'medium',
            'text': f"💬 Low class participation ({participation}%). Ask at least 1 question per lecture, join discussions, and participate in group activities."
        })
    
    # Identify strengths
    strong_subjects = [tp for tp in topic_performances if tp.score >= 80]
    if strong_subjects:
        best_subject = max(strong_subjects, key=lambda x: x.score)
        recommendations.append({
            'subject': best_subject.subject.name,
            'priority': 'low',
            'text': f"🌟 Excellent work in {best_subject.subject.name} ({best_subject.score}%)! Consider tutoring other students or working on advanced projects in this area."
        })
    
    return recommendations

# ============ AUTHENTICATION ROUTES ============

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            
            # Redirect based on role
            if user.role == 'student':
                return redirect(url_for('student_dashboard'))
            else:
                return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return render_template('register.html')
        
        # Create new user
        user = User(username=username, email=email, role=role)
        user.set_password(password)
        
        # If student, require additional info
        if role == 'student':
            user.student_id = request.form.get('student_id')
            user.name = request.form.get('name')
            user.course = request.form.get('course')
            user.year_of_study = request.form.get('year_of_study')
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ============ PASSWORD RESET ROUTES ============

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Generate reset token
            token = serializer.dumps(user.email, salt='password-reset-salt')
            
            # Create reset URL
            reset_url = url_for('reset_password', token=token, _external=True)
            
            # Send email
            try:
                msg = Message(
                    'Password Reset Request - Student Performance Tracker',
                    recipients=[user.email]
                )
                msg.html = f'''
                <html>
                <body style="font-family: Arial, sans-serif; padding: 20px;">
                    <h2>Password Reset Request</h2>
                    <p>Hi {user.name or user.username},</p>
                    <p>You requested to reset your password for the Student Performance Tracker.</p>
                    <p>Click the button below to reset your password:</p>
                    <p style="margin: 30px 0;">
                        <a href="{reset_url}" 
                           style="background-color: #007bff; color: white; padding: 12px 24px; 
                                  text-decoration: none; border-radius: 5px; display: inline-block;">
                            Reset Password
                        </a>
                    </p>
                    <p>Or copy and paste this link into your browser:</p>
                    <p style="color: #666; word-break: break-all;">{reset_url}</p>
                    <p><strong>This link will expire in 1 hour.</strong></p>
                    <p>If you didn't request this, please ignore this email.</p>
                    <hr style="margin: 30px 0;">
                    <p style="color: #999; font-size: 12px;">
                        Student Performance Tracker<br>
                        This is an automated email, please do not reply.
                    </p>
                </body>
                </html>
                '''
                mail.send(msg)
                flash('Password reset instructions have been sent to your email!', 'success')
            except Exception as e:
                flash(f'Error sending email: {str(e)}. Please contact administrator.', 'danger')
        else:
            # Don't reveal if email exists or not (security)
            flash('If that email exists, password reset instructions have been sent!', 'info')
        
        return redirect(url_for('login'))
    
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    try:
        # Verify token (valid for 1 hour)
        email = serializer.loads(token, salt='password-reset-salt', max_age=3600)
    except SignatureExpired:
        flash('The password reset link has expired. Please request a new one.', 'danger')
        return redirect(url_for('forgot_password'))
    except BadSignature:
        flash('Invalid password reset link.', 'danger')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('reset_password.html', token=token)
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long!', 'danger')
            return render_template('reset_password.html', token=token)
        
        # Update password
        user = User.query.filter_by(email=email).first()
        if user:
            user.set_password(password)
            db.session.commit()
            flash('Your password has been reset successfully! You can now login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('User not found.', 'danger')
            return redirect(url_for('forgot_password'))
    
    return render_template('reset_password.html', token=token)

# ============ MAIN ROUTES ============

@app.route('/')
@login_required
def home():
    # Students go to their dashboard
    if current_user.role == 'student':
        return redirect(url_for('student_dashboard'))
    
    # Admin/Lecturer see overview
    return render_template('home.html')

# ============ STUDENT DASHBOARD ============

@app.route('/student/dashboard')
@login_required
def student_dashboard():
    if current_user.role != 'student':
        flash('Access denied', 'danger')
        return redirect(url_for('home'))
    
    # Get student's progress entries
    progress_entries = ProgressEntry.query.filter_by(
        student_user_id=current_user.id
    ).order_by(ProgressEntry.week_number.desc()).all()
    
    # Get latest entry
    latest_entry = progress_entries[0] if progress_entries else None
    
    # Calculate statistics
    if latest_entry:
        recommendations = Recommendation.query.filter_by(
            progress_entry_id=latest_entry.id
        ).all()
    else:
        recommendations = []
    
    return render_template('student_dashboard.html',
                         progress_entries=progress_entries,
                         latest_entry=latest_entry,
                         recommendations=recommendations)

# ============ LECTURER/ADMIN: ADD PROGRESS ENTRY ============

@app.route('/progress/add/<int:student_id>', methods=['GET', 'POST'])
@login_required
def add_progress(student_id):
    if current_user.role not in ['admin', 'lecturer']:
        flash('Access denied', 'danger')
        return redirect(url_for('home'))
    
    student = User.query.filter_by(id=student_id, role='student').first_or_404()
    subjects = Subject.query.all()
    
    if request.method == 'POST':
        try:
            # Create progress entry
            week_number = int(request.form['week_number'])
            semester = request.form['semester']
            attendance = float(request.form['attendance'])
            participation = float(request.form['participation'])
            
            progress_entry = ProgressEntry(
                student_user_id=student_id,
                week_number=week_number,
                semester=semester,
                attendance_percentage=attendance,
                participation_score=participation
            )
            db.session.add(progress_entry)
            db.session.flush()  # Get ID without committing
            
            # Add topic performances
            topic_scores = []
            for subject in subjects:
                score_key = f'score_{subject.id}'
                if score_key in request.form:
                    score = float(request.form[score_key])
                    topic_scores.append(score)
                    
                    topic_perf = TopicPerformance(
                        progress_entry_id=progress_entry.id,
                        subject_id=subject.id,
                        score=score
                    )
                    db.session.add(topic_perf)
            
            # Calculate overall score and risk
            avg_topic_score = sum(topic_scores) / len(topic_scores) if topic_scores else 0
            risk_level, overall_score, _ = calculate_risk_level(attendance, avg_topic_score, participation)
            
            progress_entry.overall_score = overall_score
            progress_entry.risk_level = risk_level
            
            # Generate smart recommendations
            topic_performances = TopicPerformance.query.filter_by(progress_entry_id=progress_entry.id).all()
            smart_recs = generate_smart_recommendations(student, topic_performances, attendance, participation)
            
            for rec in smart_recs:
                recommendation = Recommendation(
                    progress_entry_id=progress_entry.id,
                    subject_id=Subject.query.filter_by(name=rec['subject']).first().id if rec['subject'] != 'General' else None,
                    recommendation_text=rec['text'],
                    priority=rec['priority']
                )
                db.session.add(recommendation)
            
            db.session.commit()
            flash(f'Progress entry added for {student.name}', 'success')
            return redirect(url_for('view_students'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'danger')
    
    return render_template('add_progress.html', student=student, subjects=subjects)

# ============ VIEW ALL STUDENTS (Lecturer/Admin) ============

@app.route('/students')
@login_required
def view_students():
    if current_user.role not in ['admin', 'lecturer']:
        flash('Access denied', 'danger')
        return redirect(url_for('home'))
    
    students = User.query.filter_by(role='student').all()
    
    # Get latest progress for each student
    student_data = []
    for student in students:
        latest_progress = ProgressEntry.query.filter_by(
            student_user_id=student.id
        ).order_by(ProgressEntry.week_number.desc()).first()
        
        student_data.append({
            'student': student,
            'latest_progress': latest_progress
        })
    
    return render_template('view_students.html', student_data=student_data)

# ============ VIEW STUDENT DETAIL (Lecturer/Admin) ============

@app.route('/student/<int:student_id>')
@login_required
def student_detail(student_id):
    if current_user.role not in ['admin', 'lecturer']:
        flash('Access denied', 'danger')
        return redirect(url_for('home'))
    
    student = User.query.filter_by(id=student_id, role='student').first_or_404()
    progress_entries = ProgressEntry.query.filter_by(
        student_user_id=student_id
    ).order_by(ProgressEntry.week_number.asc()).all()
    
    return render_template('student_detail.html', student=student, progress_entries=progress_entries)

# ============ MANAGE SUBJECTS (Admin only) ============

@app.route('/subjects', methods=['GET', 'POST'])
@login_required
def manage_subjects():
    if current_user.role != 'admin':
        flash('Access denied', 'danger')
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        name = request.form['name']
        code = request.form['code']
        description = request.form.get('description', '')
        
        subject = Subject(name=name, code=code, description=description)
        db.session.add(subject)
        db.session.commit()
        flash('Subject added successfully', 'success')
    
    subjects = Subject.query.all()
    return render_template('manage_subjects.html', subjects=subjects)

@app.route('/subjects/delete/<int:id>')
@login_required
def delete_subject(id):
    if current_user.role != 'admin':
        flash('Access denied', 'danger')
        return redirect(url_for('home'))
    
    subject = Subject.query.get_or_404(id)
    db.session.delete(subject)
    db.session.commit()
    flash('Subject deleted', 'success')
    return redirect(url_for('manage_subjects'))

# ============ DELETE STUDENT (Admin/Lecturer) ============

@app.route('/student/delete/<int:student_id>')
@login_required
def delete_student(student_id):
    if current_user.role not in ['admin', 'lecturer']:
        flash('Access denied', 'danger')
        return redirect(url_for('home'))
    
    student = User.query.filter_by(id=student_id, role='student').first_or_404()
    
    try:
        # Delete all related progress entries and their data
        progress_entries = ProgressEntry.query.filter_by(student_user_id=student_id).all()
        
        for entry in progress_entries:
            # Delete recommendations
            Recommendation.query.filter_by(progress_entry_id=entry.id).delete()
            # Delete topic performances
            TopicPerformance.query.filter_by(progress_entry_id=entry.id).delete()
            # Delete progress entry
            db.session.delete(entry)
        
        # Delete the student
        student_name = student.name
        db.session.delete(student)
        db.session.commit()
        
        flash(f'Student {student_name} and all related data deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting student: {str(e)}', 'danger')
    
    return redirect(url_for('view_students'))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))