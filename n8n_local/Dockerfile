# Use the official n8n image from Docker Hub
FROM n8nio/n8n:latest

# Set environment variables
ENV N8N_BASIC_AUTH_ACTIVE=true \
    N8N_BASIC_AUTH_USER=admin \
    N8N_BASIC_AUTH_PASSWORD=admin123 \
    N8N_HOST=localhost \
    N8N_PORT=5678 \
    VID=Showen \
    N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true

# Expose the port n8n runs on
EXPOSE 5678

# Use the default entrypoint provided by the image
CMD ["start"]
