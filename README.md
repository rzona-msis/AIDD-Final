# Campus Resource Hub

**AI-Driven Development (AiDD) - 2025 Capstone Project**  
Indiana University - Master of Science in Information Systems (MSIS)

A full-stack web application enabling university departments, student organizations, and individuals to list, share, and reserve campus resources.

## 🎯 Quick Start

```bash
# Clone and setup
git clone https://github.com/rzona-msis/AIDD-Final.git
cd "AIDD-Final"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Initialize database and run
python run.py init-db
python run.py

# Access at http://localhost:5000
```

## ✨ Key Features
- 🔍 Search & filter resources by category, location, availability
- 📅 Calendar-based booking with conflict detection
- 👥 Role-based access (Student, Staff, Admin)
- ⭐ Ratings & reviews system
- 💬 Messaging between users
- ♿ **WCAG 2.1 AA Accessibility** - Full keyboard navigation, screen reader support, ARIA labels

## 📁 Project Structure
```
app/
├── controllers/     # Flask routes (MVC)
├── models/         # Database models
├── views/          # Jinja2 templates
├── data_access/    # CRUD operations (DAL)
└── static/         # CSS, JS, images
docs/              # PRD, wireframes, ER diagram
.prompt/           # AI development log
tests/             # pytest test suite
```

See full documentation in [docs/](docs/) folder.

**Due:** November 15, 2025 | **Status:** 🚧 In Development
