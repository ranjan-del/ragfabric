# Licensing

## The project license: Apache License 2.0

RagFabric is licensed under the Apache License, Version 2.0. The full text is in [LICENSE](../LICENSE)
and the copyright notice in [NOTICE](../NOTICE).

| What you may do | Condition |
|---|---|
| Use it commercially, inside a company, for any purpose | None |
| Modify it, including the UI, providers and stores | Keep the license and copyright notices in the files you redistribute |
| Distribute it, or a modified version, in source or binary form | Include a copy of the license and the NOTICE file; state significant changes you made |
| Offer it as a hosted service | None. Apache 2.0 is not a copyleft or "source available" license |
| Combine it with proprietary code | Yes. Your code stays yours; only the RagFabric files stay under Apache 2.0 |
| Use the name RagFabric for your own product | No trademark rights are granted, apart from describing the origin of the software |

| What the license gives you beyond permission | Why it matters |
|---|---|
| An explicit patent license from every contributor | Nobody who contributed can later sue you for patent infringement over their contribution |
| A patent retaliation clause | If you sue over patents in the project, your patent license ends |
| No warranty | The software is provided as is |

### Why Apache 2.0 rather than MIT or a copyleft license

| Option | Why not |
|---|---|
| MIT | Permissive, but has no explicit patent grant. Company legal teams regularly flag this for infrastructure software |
| GPL or AGPL | Would force adopters who modify and host RagFabric to publish their changes. The project wants companies to adapt it freely, including privately |
| Business Source or similar "source available" licenses | Not open source. Restricts hosting and competes with adopters |

Apache 2.0 is the license used by Kubernetes, Airflow, Kafka and most infrastructure projects that
companies adopt. Decision recorded in [ADR 0005](adr/0005-apache-2-license.md).

## Contributor License Agreement

Contributors sign the [Individual CLA](../CLA.md) once, by posting a sentence on their first pull
request. A GitHub Action checks every pull request and blocks review until all authors have signed.
Signatures are stored in the `cla-signatures` branch of this repository.

| Question | Answer |
|---|---|
| Why a CLA when the license already covers contributions | The CLA makes the copyright and patent grants explicit, confirms the contributor has the right to contribute (for example that their employer agrees), and lets the project prove provenance if a company's legal team asks |
| Does the CLA take my copyright | No. You keep copyright. You grant a license, the same one the project grants everyone |
| Can the maintainers relicense my contribution | Only under the terms of the CLA, which ties the grant to Apache 2.0. A future change to a more restrictive license would need a new agreement |
| Corporate contributors | Ask your employer to confirm in writing that you may sign, or open an issue and a corporate CLA can be added |

The CLA text is modelled on the Apache Software Foundation Individual Contributor License Agreement.

## Third party code

Dependencies keep their own licenses. All current dependencies are permissive (MIT, BSD, Apache 2.0,
ISC). Copyleft dependencies are not accepted without discussion, because they would change the terms
adopters receive.
