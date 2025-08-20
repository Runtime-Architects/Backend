#!/bin/bash

# Domain name
DOMAIN="sustainable-city-backend.bbfpfnbpd4gmcghb.westeurope.azurecontainer.io"

# Start Python app in background first
cd /app
python source/main.py &
APP_PID=$!

# Wait for Python app to start
sleep 5

# Start nginx with self-signed cert initially
nginx

# Wait a bit more for nginx to be ready
sleep 10

# Check if Let's Encrypt certificate already exists
if [ ! -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    echo "Obtaining SSL certificate from Let's Encrypt..."
    
    # Try to get Let's Encrypt certificate
    certbot certonly \
        --webroot \
        --webroot-path=/var/www/letsencrypt \
        --email email_id@gmail.com \ ### Update Email Here
        --agree-tos \
        --no-eff-email \
        --non-interactive \
        -d $DOMAIN

    # Check if certificate was obtained successfully
    if [ -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
        echo "Certificate obtained successfully. Updating nginx config..."
        
        # Update nginx to use Let's Encrypt certificate
        sed -i "s|ssl_certificate /etc/ssl/certs/selfsigned.crt;|ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;|g" /etc/nginx/nginx.conf
        sed -i "s|ssl_certificate_key /etc/ssl/private/selfsigned.key;|ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;|g" /etc/nginx/nginx.conf
        
        # Test nginx config and reload
        nginx -t && nginx -s reload
        echo "SSL certificate installed and nginx reloaded."
    else
        echo "Failed to obtain SSL certificate. Using self-signed certificate."
    fi
else
    echo "Let's Encrypt certificate already exists. Updating nginx config..."
    # Update nginx config to use existing certificate
    sed -i "s|ssl_certificate /etc/ssl/certs/selfsigned.crt;|ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;|g" /etc/nginx/nginx.conf
    sed -i "s|ssl_certificate_key /etc/ssl/private/selfsigned.key;|ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;|g" /etc/nginx/nginx.conf
    nginx -t && nginx -s reload
fi

# Set up certificate renewal cron job
echo "0 12 * * * /usr/bin/certbot renew --quiet && /usr/sbin/nginx -s reload" | crontab -
service cron start

# Keep the container running and wait for the Python app
wait $APP_PID