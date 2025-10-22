pipeline {
    agent any

    stages {
        stage('Pull latest code') {
            steps {
                echo '📥 Pulling latest code from GitHub...'
                sh '''
                cd /var/www/brain-tumor
                git pull origin main
                '''
            }
        }

        stage('Install dependencies') {
            steps {
                echo '📦 Installing Python dependencies...'
                sh '''
                cd /var/www/brain-tumor
                . venv/bin/activate
                pip install -r requirements.txt
                deactivate
                '''
            }
        }

        stage('Restart Flask App') {
            steps {
                echo '🚀 Restarting Flask (Gunicorn) on port 8000...'
                sh '''
                cd /var/www/brain-tumor
                source venv/bin/activate
                pkill -f "gunicorn" || true
                nohup gunicorn --workers 3 --bind 0.0.0.0:8000 app:app > gunicorn.log 2>&1 &
                deactivate
                '''
            }
        }
    }
}
