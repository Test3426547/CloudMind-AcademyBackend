from fastapi_amis_admin.admin.settings import Settings
from fastapi_amis_admin.admin.site import AdminSite

# Configure Amis Admin settings
settings = Settings(database_url_async="sqlite+aiosqlite:///amisadmin.db")

# Create AdminSite instance
site = AdminSite(settings=settings)

# You can add more configuration for Amis Admin here if needed
