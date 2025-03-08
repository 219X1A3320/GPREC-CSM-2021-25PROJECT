import MySQLdb
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, session, url_for
from sklearn.preprocessing import MinMaxScaler
import pickle
import re
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)  # Fixed Flask Initialization
app.secret_key = "your_secret_key"

# Load ML models
drug = pickle.load(open('models/drug.pkl', 'rb'))
dosage = pickle.load(open('models/dosage.pkl', 'rb'))
side = pickle.load(open('models/side.pkl', 'rb'))

# MySQL Database Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '219X1A3320'
app.config['MYSQL_DB'] = 'drug'

def get_db_connection():
    return MySQLdb.connect(
        host="localhost",
        user="root",
        password="219X1A3320",
        database="drug",
        autocommit=True
    )

# ---------- Routes ----------
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    if 'loggedin' in session:
        return render_template('dashboard.html', username=session['username'])
    else:
        return redirect(url_for('login'))


@app.route('/profile')
def profile():
    if 'loggedin' in session:
        conn = get_db_connection()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM people WHERE username = %s', (session['username'],))
        account = cursor.fetchone()
        cursor.close()
        conn.close()
        return render_template('profile.html', account=account)
    else:
        return redirect(url_for('login'))


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    flash("You have been logged out successfully!", "info")
    return redirect(url_for('index'))


@app.route('/tracker')
def tracker():
    if 'loggedin' in session:
        return render_template('tracker.html')
    else:
        flash("Please login first to access the tracker!", "warning")
        return redirect(url_for('login'))


@app.route('/pharmacies')
def pharmacies():
    if 'loggedin' in session:
        return render_template('pharmacies.html')
    else:
        return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        print("Form Data:", request.form)
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM people WHERE username = %s', (username,))
        account = cursor.fetchone()
        cursor.close()
        conn.close()

        if account:
            stored_password = account['password']  # 🔹 FETCH HASHED PASSWORD
            if check_password_hash(stored_password, password):  # 🔹 COMPARE HASHED PASSWORD
                session['loggedin'] = True
                session['username'] = account['username']
                print("Session:", session)  # Debugging
                return redirect(url_for('dashboard'))
            else:
                flash('Incorrect password!', 'danger')
        else:
            flash('Username not found!', 'danger')

    return render_template('login.html', msg=msg)



from werkzeug.security import generate_password_hash, check_password_hash

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        print("Form Data:", request.form)
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        age = request.form['age']

        conn = get_db_connection()
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)

        cursor.execute('SELECT * FROM people WHERE username = %s', (username,))
        account = cursor.fetchone()

        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'^[A-Za-z0-9]+$', username):
            msg = 'Username must contain only letters and numbers!'
        else:
            hashed_password = generate_password_hash(password)  # 🔹 HASH PASSWORD
            try:
                cursor.execute('INSERT INTO people (username, password, email, age) VALUES (%s, %s, %s, %s)',
                               (username, hashed_password, email, age))
                conn.commit()
                flash('Successfully registered! Please login.', 'success')
                return redirect(url_for('login'))
            except MySQLdb.Error as e:
                msg = f"MySQL Error: {e}"
            finally:
                cursor.close()
                conn.close()

    return render_template('register.html', msg=msg)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            print("Form Data:", request.form)  # Debugging

            # Ensure request.form is a dictionary
            if not isinstance(request.form, dict):
                return render_template('prediction.html', prediction_text="Error: Invalid form data.")

            bp = request.form.get('bloodpresssure')  # Fix: Ensure form fields exist
            sugar = request.form.get('sugar')
            temperature = request.form.get('Temperature')
            age = request.form.get('age')
            condition = request.form.get('Condition')

            # Validate inputs
            if not all([bp, sugar, temperature, age, condition]) or condition == "Choose your symptom":
                return render_template('prediction.html', prediction_text="Error: Please fill all fields.")

            # Convert categorical inputs to numeric
            bp_mapping = {"Normal": 1, "Abnormal": 2}
            sugar_mapping = {"Normal": 1, "Abnormal": 2}
            condition_mapping = {
                "Depression": 1, "Lymphocytic Colitis": 2, "Urinary Tract Infection": 3, "Weight Loss": 4,
                "Birth Control": 5, "Vaginal Yeast Infection": 6, "Narcolepsy": 7, "Insomnia": 8,
                "Bipolar Disorder": 9, "Hyperhidrosis": 10, "Panic Disorder": 11, "Rosacea": 12,
                "Bowel Preparation": 13, "Constipation, Drug Induced": 14, "Diabetes, Type 2": 15,
                "Pain": 16, "Alcohol Dependence": 17, "Emergency Contraception": 18,
                "Major Depressive Disorder": 19, "Anxiety": 20, "Acne": 21, "Cough and Nasal Congestion": 22,
                "Pain and Constipation, Drug Induced": 23, "Acne and Pain": 24, "Cough, Cold and Fever": 25, "Fever": 26
            }

            # Convert categorical values
            bp = bp_mapping.get(bp)
            sugar = sugar_mapping.get(sugar)
            condition = condition_mapping.get(condition)

            # Convert numerical values safely
            temperature = float(temperature)
            age = int(age)

            # Ensure input is 2D for the model
            input_data = np.array([[bp, sugar, temperature, age, condition]])

            # Make model predictions
            drug_pred = drug.predict(input_data)
            dosage_pred = dosage.predict(input_data)
            side_pred = side.predict(input_data)

            # Format dosage
            dose = str(round(dosage_pred[0], 2))

            # Display result
            label = f"Drug: {drug_pred[0]}, Dosage: {dose} mg, Side-effects: {side_pred[0]}"
            return render_template('prediction.html', prediction_text=label)

        except Exception as e:
            return render_template('prediction.html', prediction_text=f"Error: {str(e)}")

    return render_template('prediction.html')



if __name__ == "__main__":
    app.run(debug=True)
