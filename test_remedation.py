def test_blastradius_trigger():
    # Intentionally failing to test the AI Remediation Agent
    expected_value = "Symbol_A"
    actual_result = "Symbol_B"
    
    # This assertion will fail and trigger the webhook
    assert actual_result == expected_value, "Intentional CI failure for BlastRadius!"