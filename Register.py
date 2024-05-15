import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import hashlib
import webbrowser


# Function to hash password
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Function to check hashed password
def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Function to create user table
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

# Function to add user data
def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username, password))
    conn.commit()

# Function to login user
def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username, password))
    data = c.fetchall()
    return data

with st.sidebar:
    selected = option_menu(
        menu_title="Start Here!",
        options=["Signup","Login"],
        icons=["box-seam-fill","box-seam-fill"],
        menu_icon="home",
        default_index=0
    )

if selected == "Signup":
    st.title(":iphone: :blue[Create New Account]")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password",type='password')
    if st.button("Signup"):
        create_usertable()
        add_userdata(new_user, make_hashes(new_password))
        st.success("You have successfully created a valid Account")
        st.info("Go to Login Menu to login")

elif selected == "Login":
    st.title(":calling: :blue[Login Section]")
    username = st.text_input("User Name")
    password = st.text_input("Password",type='password')
    if st.button("Login"):
        create_usertable()
        hashed_pswd = make_hashes(password)
        result = login_user(username, check_hashes(password, hashed_pswd))
        if result:
            st.success("Logged In as {}".format(username))
            st.warning("Go to Dashboard!")
            st.write(f'''
                <a target="_self" href="http://localhost:8501/">
                    <button>
                        Home
                    </button>
                </a>
                ''',
                unsafe_allow_html=True
            )

        else:
            st.warning("Incorrect Username/Password")

# Redirecting to localhost:8502
if selected == "Login":
    st.markdown('<script>window.location.href="https://www.google.com";</script>',unsafe_allow_html=True)
