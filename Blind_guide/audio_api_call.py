import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate("blind-guide-bb26d-firebase-adminsdk-1iyxl-18c9e955c8.json")
firebase_admin.initialize_app(cred)

# Get a reference to the Firestore database
db = firestore.client()

def get_ip6():
    """
    Function to get the IP6 address from Firestore.

    Parameters:
    - collection_name (str): The name of the collection.
    - document_id (str): The ID of the document to fetch.

    Returns:
    - str: The IP6 address if found, otherwise None.
    """
    collection_name = 'ip6'  # Replace with your collection name
    document_id = 'ip6' 
    try:
        # Get the document
        doc_ref = db.collection(collection_name).document(document_id)
        doc = doc_ref.get()

        # Check if the document exists
        if doc.exists:
            return doc.to_dict().get('ip6', None)  # Get the IP6 field value
        else:
            print(f"Document with ID '{document_id}' does not exist.")
            return None
    except Exception as e:
        print(f"Error getting IP6 address: {e}")
        return None

ip6_address = get_ip6()
if ip6_address:
    print(f"IP6 Address: {ip6_address}")
else:
    print("IP6 Address not found.")