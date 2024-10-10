import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v1"
MOCK_TOKEN = "mock_token_for_testing"

def test_certificate_functionality():
    print("Testing Certificate Functionality with Mock Blockchain Integration")

    headers = {"Authorization": f"Bearer {MOCK_TOKEN}"}

    # 1. Create a new certificate
    print("\n1. Creating a new certificate")
    create_certificate_url = f"{BASE_URL}/certificates"
    create_certificate_data = {
        "course_id": "course123",
        "user_id": "user456",
        "issue_date": datetime.now().isoformat()
    }
    create_certificate_response = requests.post(create_certificate_url, json=create_certificate_data, headers=headers)
    
    if create_certificate_response.status_code == 200:
        certificate = create_certificate_response.json()
        print(f"Created certificate: {json.dumps(certificate, indent=2)}")
    else:
        print(f"Failed to create certificate. Status code: {create_certificate_response.status_code}")
        print(f"Response: {create_certificate_response.text}")
        return

    # 2. Retrieve the created certificate
    print("\n2. Retrieving the created certificate")
    get_certificate_url = f"{BASE_URL}/certificates/{certificate['id']}"
    get_certificate_response = requests.get(get_certificate_url, headers=headers)
    
    if get_certificate_response.status_code == 200:
        retrieved_certificate = get_certificate_response.json()
        print(f"Retrieved certificate: {json.dumps(retrieved_certificate, indent=2)}")
    else:
        print(f"Failed to retrieve certificate. Status code: {get_certificate_response.status_code}")

    # 3. Verify the certificate
    print("\n3. Verifying the certificate")
    verify_certificate_url = f"{BASE_URL}/certificates/{certificate['id']}/verify"
    verify_certificate_data = {"certificate_hash": certificate['hash']}
    verify_certificate_response = requests.post(verify_certificate_url, json=verify_certificate_data, headers=headers)
    
    if verify_certificate_response.status_code == 200:
        verification_result = verify_certificate_response.json()
        print(f"Verification result: {json.dumps(verification_result, indent=2)}")
    else:
        print(f"Failed to verify certificate. Status code: {verify_certificate_response.status_code}")

    # 4. Revoke the certificate
    print("\n4. Revoking the certificate")
    revoke_certificate_url = f"{BASE_URL}/certificates/{certificate['id']}/revoke"
    revoke_certificate_response = requests.post(revoke_certificate_url, headers=headers)
    
    if revoke_certificate_response.status_code == 200:
        revocation_result = revoke_certificate_response.json()
        print(f"Revocation result: {json.dumps(revocation_result, indent=2)}")
    else:
        print(f"Failed to revoke certificate. Status code: {revoke_certificate_response.status_code}")

    # 5. Try to verify the revoked certificate
    print("\n5. Attempting to verify the revoked certificate")
    verify_revoked_response = requests.post(verify_certificate_url, json=verify_certificate_data, headers=headers)
    
    if verify_revoked_response.status_code == 200:
        verification_result = verify_revoked_response.json()
        print(f"Verification result for revoked certificate: {json.dumps(verification_result, indent=2)}")
    else:
        print(f"Failed to verify revoked certificate. Status code: {verify_revoked_response.status_code}")

if __name__ == "__main__":
    test_certificate_functionality()
