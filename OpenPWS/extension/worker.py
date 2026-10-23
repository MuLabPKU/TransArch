"""Worker-side patch installation; real inference is armed by the controller."""

from .patch import AuditWorker, install

install("two_stage")


class Worker(AuditWorker):
    pass
