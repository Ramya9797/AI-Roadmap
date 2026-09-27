class HumanApproval:
    VALID_STATUSES = {
        "pending",
        "approved",
        "rejected",
    }

    def __init__(self):
        self.status = "pending"

    def set_status(self, status):
        if status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid approval status: {status}"
            )

        self.status = status

    def approve(self):
        self.set_status("approved")

    def reject(self):
        self.set_status("rejected")