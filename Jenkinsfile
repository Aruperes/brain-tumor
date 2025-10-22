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
                source venv/bin/activate
                pip install -r requirements.txt
                deactivate
                '''
            }
        }

        stage('Restart Flask App') {
            steps {
                echo '🚀 Restarting Flask on port 8000...'
                sh '''
                cd /var/www/brain-tumor
                source venv/bin/activate

                # Stop any existing Flask process
                pkill -f "python app.py" || true

                # Start Flask again in background
                nohup python app.py > flask.log 2>&1 &

                deactivate
                '''
            }
        }
    }
}
