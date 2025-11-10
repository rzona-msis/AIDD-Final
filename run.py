"""
Flask Application Entry Point
Campus Resource Hub - AiDD 2025 Capstone

This file initializes and runs the Flask application.
"""

import os
import sys
from app import create_app, db
from app.models.user import User
from app.models.resource import Resource
from app.models.booking import Booking

# Create Flask app
app = create_app()

@app.cli.command()
def init_db():
    """Initialize the database with tables."""
    with app.app_context():
        db.create_all()
        print("✅ Database initialized successfully!")
        
        # Create default admin user if none exists
        admin = User.query.filter_by(email='admin@campus.edu').first()
        if not admin:
            from werkzeug.security import generate_password_hash
            admin = User(
                name='System Administrator',
                email='admin@campus.edu',
                password_hash=generate_password_hash('admin123'),  # Change in production!
                role='admin',
                department='IT'
            )
            db.session.add(admin)
            db.session.commit()
            print("✅ Default admin user created (admin@campus.edu / admin123)")

@app.cli.command()
def reset_db():
    """Drop and recreate all database tables."""
    with app.app_context():
        db.drop_all()
        db.create_all()
        print("✅ Database reset successfully!")

if __name__ == '__main__':
    # Check if init-db command
    if len(sys.argv) > 1 and sys.argv[1] == 'init-db':
        with app.app_context():
            db.create_all()
            print("✅ Database initialized!")
    else:
        # Run the Flask development server
        app.run(debug=True, host='0.0.0.0', port=5000)
