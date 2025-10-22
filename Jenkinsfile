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
                    bash -c "source venv/bin/activate && pip install -r requirements.txt"
                '''
            }
        }

        stage('Restart Flask App') {
            steps {
                echo '🚀 Restarting Flask on port 8000...'
                sh '''
                    cd /var/www/brain-tumor
                    bash -c "
                        source venv/bin/activate && \
                        pkill -f 'python app.py' || true && \
                        nohup python app.py --port=8000 > flask.log 2>&1 &
                    "
                '''
            }
        }
    }
}
