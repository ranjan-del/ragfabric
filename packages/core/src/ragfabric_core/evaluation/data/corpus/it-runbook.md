# IT Operations Runbook

This runbook lists the error codes raised by Brightwater's internal platform and how to resolve them. Escalate anything not listed here to the IT Operations team lead.

## Error codes

BW-7731: The nightly warehouse export timed out. Rerun the export job with the --resume flag; if it fails twice, page the Data Platform on-call engineer.

BW-7732: The export job could not authenticate to the storage cluster. Rotate the service credential in the secrets vault and rerun the job.

BW-4410: A user's laptop failed disk encryption checks. Reimage the laptop within 2 working days; the user must not connect to the VPN until it is fixed.

BW-5120: The VPN rejected a certificate that has expired. Issue a new device certificate from the IT portal; certificates are valid for 365 days.

## On-call

The IT Operations on-call rotation changes every Monday at 09:00. Ivan Petrov maintains the rota.
