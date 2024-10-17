from setuptools import setup, find_packages

setup(
    name="cloudmind-academy-backend",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "new_venv*"]),
    install_requires=[
        "fastapi>=0.95.1,<0.100.0",
        "pydantic>=1.10.7,<2.0.0",
        "web3>=5.31.3,<6.0.0",
        "eth-account>=0.5.9,<0.6.0",
        "supabase>=2.9.0,<3.0.0",
        "websockets>=9.1,<10.0",
        # Add other dependencies here, but remove conflicting ones
    ],
)
