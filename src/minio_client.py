import boto3
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import json
import re
import hashlib

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class MinIOClient:
    def __init__(self):
        """
        Initialize MinIO client
        Read configuration from environment variables:
        - MINIO_ENDPOINT: MinIO server address
          * RAGFlow built-in: http://minio:9000
          * Standalone: http://localhost:9000
        - MINIO_ACCESS_KEY: Access key (RAGFlow default: rag_flow)
        - MINIO_SECRET_KEY: Secret key (RAGFlow default: infini_rag_flow)
        - MINIO_BUCKET_NAME: Bucket name (optional, default uses document name)
        """
        self.endpoint = os.getenv('MINIO_ENDPOINT', '').rstrip('/')
        self.public_base_url = os.getenv('MINIO_PUBLIC_BASE_URL', self.endpoint).rstrip('/')
        self.access_key = os.getenv('MINIO_ACCESS_KEY')
        self.secret_key = os.getenv('MINIO_SECRET_KEY')
        self.bucket_name = os.getenv('MINIO_BUCKET_NAME', 'ragflow-images')

        if not all([self.endpoint, self.access_key, self.secret_key]):
            raise ValueError("Missing MinIO configuration. Please set MINIO_ENDPOINT, MINIO_ACCESS_KEY, and MINIO_SECRET_KEY in the .env file")

        # Create S3 client (MinIO compatible)
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name='us-east-1'  # MinIO doesn't need a real region
        )

        print("MinIO client initialized successfully")
        print(f"Endpoint: {self.endpoint}")
        print(f"Public Base URL: {self.public_base_url}")
        print(f"Bucket: {self.bucket_name}")

    def create_bucket_if_not_exists(self, bucket_name=None):
        """
        Create bucket if it doesn't exist and set public access policy

        Parameters:
        - bucket_name: Name of the bucket. If not provided, use the default name.
        """
        bucket = bucket_name or self.bucket_name

        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=bucket)
            print(f"Bucket '{bucket}' already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Bucket doesn't exist, create it
                print(f"Creating bucket '{bucket}'...")
                try:
                    # For MinIO, specify CreateBucketConfiguration
                    self.s3_client.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={'LocationConstraint': 'us-east-1'}
                    )
                    print(f"Bucket '{bucket}' created successfully")
                except Exception as create_error:
                    print(f"Failed to create bucket: {create_error}")
                    raise
            else:
                print(f"Failed to check bucket status: {e}")
                raise

        # Set public access policy
        self.set_public_read_policy(bucket)
        return bucket

    def set_public_read_policy(self, bucket_name):
        """
        Set public read policy for the bucket, allowing everyone to read objects

        Parameters:
        - bucket_name: Name of the bucket
        """
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{bucket_name}/*"
                }
            ]
        }

        try:
            self.s3_client.put_bucket_policy(
                Bucket=bucket_name,
                Policy=json.dumps(policy)
            )
            print(f"Bucket '{bucket_name}' set to public read")
        except Exception as e:
            print(f"Failed to set public access policy: {e}")
            # Some MinIO configs might not support bucket policy; try object-level access if needed
            print("Attempting to set object-level public access...")

    def upload_image(self, file_path, object_key, bucket_name=None):
        """
        Upload image file to MinIO

        Parameters:
        - file_path: Local file path
        - object_key: Object key in MinIO (filename)
        - bucket_name: Bucket name, optional

        Returns:
        - Public access URL
        """
        bucket = bucket_name or self.bucket_name

        # Ensure bucket exists
        self.create_bucket_if_not_exists(bucket)

        try:
            # Upload file
            with open(file_path, 'rb') as file_data:
                self.s3_client.upload_fileobj(
                    file_data,
                    bucket,
                    object_key,
                    ExtraArgs={
                        'ContentType': 'image/png',
                        'ACL': 'public-read'
                    }
                )

            # Generate public access URL
            public_url = f"{self.public_base_url}/{bucket}/{object_key}"
            print(f"Image uploaded successfully: {public_url}")
            return public_url

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            print(f"Upload failed: {e}")
            raise

    def upload_image_bytes(self, image_bytes, object_key, content_type='image/png', bucket_name=None):
        """
        Directly upload image bytes to MinIO

        Parameters:
        - image_bytes: Image byte data
        - object_key: Object key in MinIO
        - content_type: Content type
        - bucket_name: Bucket name, optional

        Returns:
        - Public access URL
        """
        bucket = bucket_name or self.bucket_name

        # Ensure bucket exists
        self.create_bucket_if_not_exists(bucket)

        try:
            # Upload byte data
            self.s3_client.put_object(
                Bucket=bucket,
                Key=object_key,
                Body=image_bytes,
                ContentType=content_type,
                ACL='public-read'
            )

            # Generate public access URL
            public_url = f"{self.public_base_url}/{bucket}/{object_key}"
            print(f"Image uploaded successfully: {public_url}")
            return public_url

        except Exception as e:
            print(f"Upload failed: {e}")
            raise

    def get_bucket_url(self, bucket_name=None):
        """
        Get the base URL of the bucket

        Parameters:
        - bucket_name: Bucket name, optional

        Returns:
        - Base URL of the bucket
        """
        bucket = bucket_name or self.bucket_name
        return f"{self.public_base_url}/{bucket}"

    def list_objects(self, bucket_name=None, prefix=""):
        """
        List objects in the bucket

        Parameters:
        - bucket_name: Bucket name, optional
        - prefix: Object key prefix for filtering

        Returns:
        - List of object keys
        """
        bucket = bucket_name or self.bucket_name

        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            print(f"Failed to list objects: {e}")
            return []

# Global MinIO client instance
_minio_client = None

def get_minio_client():
    """
    Get MinIO client singleton instance

    Returns:
    - MinIOClient instance
    """
    global _minio_client
    if _minio_client is None:
        try:
            _minio_client = MinIOClient()
        except Exception as e:
            print(f"Failed to initialize MinIO client: {e}")
            return None
    return _minio_client

def init_minio_bucket(pdf_filename=None, custom_bucket_name=None):
    """
    Initialize MinIO bucket, creating a dedicated bucket for the PDF document

    Parameters:
    - pdf_filename: PDF filename used to generate bucket name
    - custom_bucket_name: Custom bucket name (optional, takes precedence)

    Returns:
    - Bucket name and base URL
    """
    client = get_minio_client()
    if not client:
        return None, None

    # Use custom bucket name if provided
    if custom_bucket_name:
        bucket_name = custom_bucket_name
    # Otherwise, generate bucket name from PDF filename if provided
    elif pdf_filename:
        base_name = os.path.splitext(os.path.basename(pdf_filename))[0]
        
        # 1. Convert to lowercase
        safe_name = base_name.lower()
        
        # 2. Replace invalid characters with hyphens (keep a-z, 0-9, ., -, _)
        safe_name = re.sub(r'[^a-z0-9.\-_]', '-', safe_name)
        
        # 3. Use MD5 hash if result is empty or just special characters
        if not safe_name.replace('-', '').replace('_', '').replace('.', ''):
            name_hash = hashlib.md5(base_name.encode('utf-8')).hexdigest()[:8]
            bucket_name = f"ragflow-{name_hash}"
        else:
            # Strip special symbols from start/end
            safe_name = safe_name.strip('.-_')
            # Avoid consecutive hyphens
            safe_name = re.sub(r'-+', '-', safe_name)
            bucket_name = f"ragflow-{safe_name}"

        # 4. Ensure length compliance (max 63 characters for safety)
        if len(bucket_name) > 63:
            bucket_name = bucket_name[:63]
            
    else:
        bucket_name = client.bucket_name

    try:
        client.create_bucket_if_not_exists(bucket_name)
        base_url = client.get_bucket_url(bucket_name)
        return bucket_name, base_url
    except Exception as e:
        print(f"Failed to initialize MinIO bucket: {e}")
        return None, None

if __name__ == "__main__":
    # Test script
    print("Testing MinIO client...")

    try:
        client = get_minio_client()
        if client:
            bucket_name, base_url = init_minio_bucket("test_document.pdf")
            print(f"Test successful! Bucket: {bucket_name}, URL: {base_url}")
        else:
            print("MinIO client initialization failed")
    except Exception as e:
        print(f"Test failed: {e}")
