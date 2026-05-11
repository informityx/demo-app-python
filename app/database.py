from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import os
from werkzeug.security import generate_password_hash

db = SQLAlchemy()


def init_db():
    """Initialize database and migrate data from Excel files"""
    try:
        from app.models.user import User
        from app.models.booking import Booking
        
        # Create all tables
        try:
            db.create_all()
        except Exception as e:
            print(f"Warning: Could not create database tables: {e}")
            # On Vercel, this might fail due to file system limitations
            return
        
        # Migrate users from Excel if table is empty
        try:
            if User.query.count() == 0:
                users_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'users.xlsx')
                if os.path.exists(users_file):
                    try:
                        df = pd.read_excel(users_file)
                        for _, row in df.iterrows():
                            user = User(
                                username=row['username'],
                                password=generate_password_hash(str(row['password'])),
                                role=row.get('role', 'Employee')
                            )
                            db.session.add(user)
                        db.session.commit()
                        print(f"Migrated {len(df)} users from Excel")
                    except Exception as e:
                        print(f"Error migrating users: {e}")
                        db.session.rollback()
                else:
                    # No users.xlsx: create default admin for local dev/testing
                    try:
                        admin = User(
                            username='admin',
                            password=generate_password_hash('password123'),
                            role='Employee'
                        )
                        db.session.add(admin)
                        db.session.commit()
                        print("Created default user: admin / password123")
                    except Exception as e:
                        print(f"Error creating default user: {e}")
                        db.session.rollback()
        except Exception as e:
            print(f"Warning: User migration failed: {e}")
        
        # Migrate bookings from Excel if table is empty
        try:
            if Booking.query.count() == 0:
                bookings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bookings.xlsx')
                if os.path.exists(bookings_file):
                    try:
                        df = pd.read_excel(bookings_file)
                        for _, row in df.iterrows():
                            booking = Booking(
                                booking_id=row.get('Booking ID', ''),
                                booking_type=row.get('Booking Type', 'Service'),
                                name=row.get('Name', ''),
                                phone=row.get('Phone', ''),
                                vehicle_model=row.get('Vehicle Model', ''),
                                preferred_date=pd.to_datetime(row.get('Preferred Date', pd.Timestamp.now())).date() if pd.notna(row.get('Preferred Date')) else None
                            )
                            db.session.add(booking)
                        db.session.commit()
                        print(f"Migrated {len(df)} bookings from Excel")
                    except Exception as e:
                        print(f"Error migrating bookings: {e}")
                        db.session.rollback()
        except Exception as e:
            print(f"Warning: Booking migration failed: {e}")
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")
        # Don't raise - allow app to continue without database initialization
        # Database will be created on first use if possible
