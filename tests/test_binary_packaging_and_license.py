import pytest
import os
from turing.core.license_gate import TuringLicenseGate, LicenseTier

def test_license_gate_single_gpu_community():
    res = TuringLicenseGate.verify_runtime_environment(total_gpus=1, is_commercial_revenue_fleet=False)
    assert res["status"] == "AUTHORIZED"
    assert res["tier"] == LicenseTier.COMMUNITY_FREE
    assert "BSL 1.1" in res["terms"]

def test_license_gate_enterprise_commercial():
    os.environ["TURING_LICENSE_KEY"] = "TURING-ENT-9948-284B-753B-PROD-CLUSTER"
    try:
        res = TuringLicenseGate.verify_runtime_environment(total_gpus=8, is_commercial_revenue_fleet=True)
        assert res["status"] == "AUTHORIZED"
        assert res["tier"] == LicenseTier.ENTERPRISE_COMMERCIAL
        assert "Enterprise" in res["terms"]
    finally:
        del os.environ["TURING_LICENSE_KEY"]

def test_license_gate_multi_gpu_without_key():
    res = TuringLicenseGate.verify_runtime_environment(total_gpus=8, is_commercial_revenue_fleet=True)
    assert res["status"] == "COMMUNITY_EVALUATION"
    assert "warning" in res

