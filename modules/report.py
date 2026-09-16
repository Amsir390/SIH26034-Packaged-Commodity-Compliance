def generate_report(compliance_result):
    """
    Prepare a simple compliance report
    from the rule engine result.
    """

    overall_status = compliance_result["overall_status"]
    checks = compliance_result["checks"]

    passed = [
        check for check in checks
        if check["status"] == "PASS"
    ]

    failed = [
        check for check in checks
        if check["status"] == "FAIL"
    ]

    report = {
        "overall_status": overall_status,
        "total_checks": len(checks),
        "passed": len(passed),
        "failed": len(failed),
        "violations": [
            check["message"] for check in failed
        ]
    }

    return report