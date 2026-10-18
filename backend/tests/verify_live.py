import os
os.environ['no_proxy'] = '*'
import urllib.request
import json
from datetime import datetime

base = 'http://127.0.0.1:8000'

def post_json(endpoint, data, token=None):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(
        base + endpoint,
        data=json.dumps(data).encode('utf-8'),
        headers=headers,
    )
    res = urllib.request.urlopen(req)
    return json.loads(res.read().decode())

def get_json(endpoint, token=None):
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(
        base + endpoint,
        headers=headers,
    )
    res = urllib.request.urlopen(req)
    return json.loads(res.read().decode())

def main():
    print('=== 1. TEST REGISTRATION & LOGIN ===')
    email = f"live_verify_{datetime.now().strftime('%Y%m%d%H%M%S%f')}@example.com"
    reg = post_json('/api/auth/register', {
        'email': email,
        'password': 'Password123!',
        'full_name': 'Demo User',
        'phone': '9123456780',
    })
    token = reg['access_token']
    print(f'[OK] Registered user: {email}')

    print('=== 2. START COMPLAINT SESSION ===')
    start = post_json('/api/chat/start', {}, token)
    cid = start['complaint_id']
    print(f'[OK] Started complaint #{cid}')

    print('=== 3. CHAT GREETING CHECK ===')
    msg1 = post_json('/api/chat/message', {'complaint_id': cid, 'message': 'hello'}, token)
    cat1 = msg1['classification']['category']
    status1 = msg1['classification'].get('status')
    print(f'[OK] Message "hello" -> Category: "{cat1}", Status: "{status1}"')
    assert cat1 == 'Pending Incident Details', f'Expected Pending Incident Details, got {cat1}'
    assert status1 == 'conversational', f'Expected conversational status, got {status1}'

    print('=== 4. GENERIC HELP PLEA CHECK ===')
    msg2 = post_json('/api/chat/message', {'complaint_id': cid, 'message': 'help me please'}, token)
    cat2 = msg2['classification']['category']
    print(f'[OK] Message "help me please" -> Category: "{cat2}"')
    assert cat2 == 'Pending Incident Details'

    print('=== 5. DETAILED CYBERCRIME CLASSIFICATION ===')
    crime_text = 'Someone hacked my WhatsApp and sent a fraudulent UPI payment link to my family members and stole 35000 rupees'
    msg3 = post_json('/api/chat/message', {'complaint_id': cid, 'message': crime_text}, token)
    cat3 = msg3['classification']['category']
    conf3 = msg3['classification']['confidence']
    print(f'[OK] Detailed text -> Authoritative Category: "{cat3}" (Confidence: {conf3:.1%})')
    assert conf3 > 0.40, f'Expected confident classification, got {conf3}'

    print('=== 6. SUBSEQUENT GREETING PRESERVATION ===')
    msg4 = post_json('/api/chat/message', {'complaint_id': cid, 'message': 'ok thank you'}, token)
    cat4 = msg4['classification']['category']
    conf4 = msg4['classification']['confidence']
    print(f'[OK] Subsequent "ok thank you" -> Preserved Category: "{cat4}" (Confidence: {conf4:.1%})')
    assert cat4 == cat3, f'Classification was overwritten! Expected {cat3}, got {cat4}'

    print('=== 7. COMPLAINT LIST & DETAIL VERIFICATION ===')
    complaints = get_json('/api/complaint/list', token)['complaints']
    target = [c for c in complaints if c['id'] == cid][0]
    print(f'[OK] List API item -> Category: "{target["authoritative_category"]}", Confidence: {target["confidence"]}')
    assert target['authoritative_category'] == cat3
    assert target['confidence'] is not None

    detail = get_json(f'/api/complaint/{cid}', token)
    print(f'[OK] Detail API -> Category: "{detail["authoritative_category"]}", Confidence: {detail["confidence"]}')
    assert detail['authoritative_category'] == cat3
    assert detail['confidence'] is not None

    print('=== 8. RISK RECALCULATION ON UPDATE ===')
    req_patch = urllib.request.Request(
        f'{base}/api/complaint/{cid}',
        data=json.dumps({'financial_loss': 60000.0, 'incident_date': '2026-10-16'}).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'},
        method='PATCH',
    )
    patch_res = json.loads(urllib.request.urlopen(req_patch).read().decode())
    print(f'[OK] Update API -> Category: "{patch_res["authoritative_category"]}", Risk Score: {patch_res["risk_score"]}')
    assert patch_res['authoritative_category'] == cat3
    assert patch_res['risk_score'] >= 50

    print('=== 9. PDF EXPORT VERIFICATION ===')
    req_pdf = urllib.request.Request(f'{base}/api/complaint/{cid}/pdf', headers={'Authorization': f'Bearer {token}'})
    pdf_data = urllib.request.urlopen(req_pdf).read()
    print(f'[OK] PDF Export -> Generated {len(pdf_data)} bytes of valid PDF')
    assert len(pdf_data) > 1000

    print('\n>>> ALL 9 END-TO-END WORKFLOW CHECKS PASSED LIVE ON RUNNING SERVER! <<<')

if __name__ == '__main__':
    main()
