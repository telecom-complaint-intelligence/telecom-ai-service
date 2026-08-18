import os
import sys
import json

# Setup path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

def main():
    cloud_kb_path = os.path.join(PROJECT_ROOT, "tests", "evaluation", "ragas", "cloud_knowledge_base.json")
    with open(cloud_kb_path, "r", encoding="utf-8") as f:
        cloud_kbs = json.load(f)
        
    golden_path = os.path.join(PROJECT_ROOT, "tests", "evaluation", "golden_dataset.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)
        
    # Map cloud KBs by ID
    cloud_map = {kb["id"]: kb for kb in cloud_kbs if "id" in kb}
    
    # Specific IDs to inspect
    target_ids = ["KB076", "KB079", "KB009", "KB059", "KB063", "KB061", "KB046", "KB047", "KB060"]
    
    print("\n==================================================")
    print("PRODUCTION CLOUD KB CONTENTS FOR REQUESTED IDS")
    print("==================================================")
    for tid in target_ids:
        kb = cloud_map.get(tid)
        if kb:
            print(f"\nID: {kb['id']}")
            print(f"Title: {kb['title']}")
            print(f"Domain: {kb['domain']}")
            print(f"Content: {kb['content']}")
            print("--------------------------------------------------")
        else:
            print(f"\nID: {tid} - NOT FOUND IN CLOUD KB")
            print("--------------------------------------------------")
            
    print("\n==================================================")
    print("COMPARISON WITH EXPECTED GOLDEN KNOWLEDGE")
    print("==================================================")
    
    comparisons = [
        {
            "case": "EVAL-001 (Account Access)",
            "expected_id": "KB_PORTAL_AUTH_01",
            "expected_title": "Resetting Customer Portal Passwords",
            "expected_content": "Verify login credentials. Click on 'Forgot Password' link to receive a password reset token. Clean browser cookies and cache to resolve temporary 403 forbidden or password mismatch errors.",
            "production_candidate_id": "KB076",
            "production_candidate_title": "Password Reset",
            "production_candidate_content": cloud_map.get("KB076", {}).get("content", "N/A")
        },
        {
            "case": "EVAL-002 (Account Access)",
            "expected_id": "KB_EMAIL_SERVER_01",
            "expected_title": "Email Delivery and Notification Troubleshooting",
            "expected_content": "Check spam folder. Verify recipient email address. Check system notification settings. Restart device or refresh mailbox. Contact support if email system shows delivery failures.",
            "production_candidate_id": "KB079",
            "production_candidate_title": "Email Verification Problem",
            "production_candidate_content": cloud_map.get("KB079", {}).get("content", "N/A")
        },
        {
            "case": "EVAL-003 & EVAL-009 (Internet Outage / Performance)",
            "expected_id": "KB_SWITCH_OUTAGE_01",
            "expected_title": "Broadband Switch and Fiber Loop Diagnostics",
            "expected_content": "Query local exchange switch status. Check fiber loop telemetry for optical power degradation. Verify backup link routing state. Perform port diagnostic reset.",
            "production_candidate_id": "KB059 / KB060 / KB047",
            "production_candidate_title": "Service Outage Status / Outage Resolved but Service Down / Multiple Devices Offline",
            "production_candidate_content": (
                f"KB059: {cloud_map.get('KB059', {}).get('content', 'N/A')}\n"
                f"KB060: {cloud_map.get('KB060', {}).get('content', 'N/A')}\n"
                f"KB047: {cloud_map.get('KB047', {}).get('content', 'N/A')}"
            )
        },
        {
            "case": "EVAL-006 (Billing Overcharge)",
            "expected_id": "HIST-1006",
            "expected_title": "Billing Overcharge Resolution",
            "expected_content": "Check duplicate payment authorization records. Reconcile customer ledger. Process credit refund for double charges within 3-5 business days.",
            "production_candidate_id": "KB009 / KB013",
            "production_candidate_title": "Duplicate Billing Charge / Autopay Failure",
            "production_candidate_content": (
                f"KB009: {cloud_map.get('KB009', {}).get('content', 'N/A')}\n"
                f"KB013: {cloud_map.get('KB013', {}).get('content', 'N/A')}"
            )
        }
    ]
    
    for comp in comparisons:
        print(f"\nCase: {comp['case']}")
        print(f"  Expected ({comp['expected_id']}): {comp['expected_title']}")
        print(f"    Content: {comp['expected_content']}")
        print(f"  Production Candidate ({comp['production_candidate_id']}): {comp['production_candidate_title']}")
        print(f"    Content: {comp['production_candidate_content']}")
        print("--------------------------------------------------")

if __name__ == "__main__":
    main()
