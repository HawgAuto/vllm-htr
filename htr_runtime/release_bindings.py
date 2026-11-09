"""Fail-closed reconciliation of explicitly changed installed bindings."""


def reconcile(previous, actual, corroborated, allowed):
    """Return updated hashes only for declared, independently bound changes."""
    if set(previous) != set(actual):
        raise ValueError('binding member set changed')
    result = dict(previous)
    for name, digest in previous.items():
        if actual[name] == digest:
            continue
        if name not in allowed or corroborated.get(name) != actual[name]:
            raise ValueError('unapproved or uncorroborated binding drift: ' + name)
        result[name] = actual[name]
    return result
