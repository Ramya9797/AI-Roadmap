from rag.human_approval import HumanApproval


def test_approval_starts_pending():
    approval = HumanApproval()

    assert approval.status == "pending"


def test_approval_can_be_granted():
    approval = HumanApproval()

    approval.approve()

    assert approval.status == "approved"


def test_approval_can_be_rejected():
    approval = HumanApproval()

    approval.reject()

    assert approval.status == "rejected"


def test_approval_rejects_invalid_status():
    approval = HumanApproval()

    try:
        approval.set_status("invalid")
        assert False
    except ValueError:
        assert True