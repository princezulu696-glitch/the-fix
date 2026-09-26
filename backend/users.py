import hashlib
import hmac
import os
import sqlite3
import uuid
from datetime import datetime


# ============================================================
# DATABASE
# ============================================================

DATABASE = "users.db"


def get_connection():
    return sqlite3.connect(DATABASE)


# ============================================================
# PASSWORD SECURITY
# ============================================================

def hash_password(password):

    salt = os.urandom(32)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        600000
    )

    return (
        salt.hex()
        + ":"
        + password_hash.hex()
    )


def verify_password(password, stored_hash):

    try:

        salt_hex, hash_hex = stored_hash.split(":")

        salt = bytes.fromhex(salt_hex)

        expected_hash = bytes.fromhex(hash_hex)

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            600000
        )

        return hmac.compare_digest(
            password_hash,
            expected_hash
        )

    except Exception:

        return False


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()

    connection.close()


# ============================================================
# CREATE USER
# ============================================================

def create_user(username, email, password):

    username = str(username).strip()

    email = str(email).strip().lower()

    password = str(password)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not username:

        raise ValueError(
            "Username cannot be empty."
        )


    if not email:

        raise ValueError(
            "Email cannot be empty."
        )


    if not password:

        raise ValueError(
            "Password cannot be empty."
        )


    if len(username) < 3:

        raise ValueError(
            "Username must be at least 3 characters."
        )


    if len(password) < 8:

        raise ValueError(
            "Password must be at least 8 characters."
        )


    if "@" not in email:

        raise ValueError(
            "Please enter a valid email address."
        )


    # --------------------------------------------------------
    # PASSWORD HASH
    # --------------------------------------------------------

    password_hash = hash_password(
        password
    )


    # --------------------------------------------------------
    # USER INFORMATION
    # --------------------------------------------------------

    user_id = (
        "user_"
        + str(uuid.uuid4())
    )


    created_at = datetime.now().isoformat()


    # --------------------------------------------------------
    # DATABASE INSERT
    # --------------------------------------------------------

    connection = get_connection()

    cursor = connection.cursor()


    try:

        cursor.execute(
            """
            INSERT INTO users
            (
                user_id,
                username,
                email,
                password_hash,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                username,
                email,
                password_hash,
                created_at
            )
        )

        connection.commit()


    except sqlite3.IntegrityError as error:

        connection.rollback()

        error_message = str(error).lower()


        if "username" in error_message:

            message = (
                "Username already exists."
            )

        elif "email" in error_message:

            message = (
                "Email already exists."
            )

        else:

            message = (
                "Username or email already exists."
            )


        raise ValueError(message)


    finally:

        connection.close()


    # --------------------------------------------------------
    # RETURN SAFE USER DATA
    # --------------------------------------------------------

    return {

        "user_id":
            user_id,

        "username":
            username,

        "email":
            email,

        "created_at":
            created_at
    }


# ============================================================
# AUTHENTICATE USER
# ============================================================

def authenticate_user(username, password):

    username = str(username).strip()

    password = str(password)


    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT
            user_id,
            username,
            email,
            password_hash,
            created_at
        FROM users
        WHERE username = ?
        """,
        (username,)
    )


    user = cursor.fetchone()

    connection.close()


    if not user:

        return None


    user_id = user[0]

    stored_username = user[1]

    email = user[2]

    password_hash = user[3]

    created_at = user[4]


    if not verify_password(
        password,
        password_hash
    ):

        return None


    return {

        "user_id":
            user_id,

        "username":
            stored_username,

        "email":
            email,

        "created_at":
            created_at
    }


# ============================================================
# GET USER
# ============================================================

def get_user(user_id):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT
            user_id,
            username,
            email,
            created_at
        FROM users
        WHERE user_id = ?
        """,
        (user_id,)
    )


    user = cursor.fetchone()

    connection.close()


    if not user:

        return None


    return {

        "user_id":
            user[0],

        "username":
            user[1],

        "email":
            user[2],

        "created_at":
            user[3]
    }


# ============================================================
# START DATABASE
# ============================================================

init_database()
