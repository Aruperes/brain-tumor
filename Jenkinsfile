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
                bash -c "
                cd /var/www/brain-tumor &&
                . venv/bin/activate &&
                pip install -r requirements.txt
                "
                '''
            }
        }

        stage('Restart Flask App') {
            steps {
                echo '🚀 Restarting Flask (Gunicorn) on port 8000...'
                sh '''
                bash -c "
                cd /var/www/brain-tumor &&
                # Hentikan proses lama di port 8000 (kalau ada)
                sudo fuser -k 8000/tcp || true &&
                . venv/bin/activate &&
                nohup gunicorn --workers 3 --bind 0.0.0.0:8000 app:app > gunicorn.log 2>&1 &
                "
                '''
            }
        }
    }
}
