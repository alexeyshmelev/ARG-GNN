"""ROOT-owned inactive authorizer body. Callable only after fresh whole-join review and instruction."""
def authorize_inline_ROOT_retirement_v1(*,guard,context,finite):
    from pathlib import Path
    from datetime import datetime, timezone, timedelta
    import hashlib, json, os, stat
    finite()
    paths=context['fresh_metadata_paths']
    def write_new(kind,value):
        finite()
        path=guard.budget.v4.canonical(guard.ROOT,paths[kind])
        data=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode('utf-8')
        if len(data)>guard.MAX_JSON:
            raise ValueError('Original compact metadata byte cap')
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            opened=os.fstat(stream.fileno())
        named=path.lstat()
        if not (stat.S_ISREG(named.st_mode) and named.st_nlink==1 and named.st_size==len(data)
                and (opened.st_dev,opened.st_ino)==(named.st_dev,named.st_ino)):
            raise ValueError('New immutable ROOT metadata pathname changed')
        directory_fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        finite()
        return {'path':str(path.relative_to(guard.ROOT)),'sha256':hashlib.sha256(data).hexdigest()}
    key=context['key']
    own=context['original_reservation']
    if own['key']!=key:
        raise ValueError('Same exact original observation reservation')
    instruction=guard._json(context['root_instruction_ref'])
    finite()
    facts=guard._observe_targets(guard._target(key))
    observation=guard.stopped_observation_draft_v1(key,revocation_ref=context['permanent_revocation_ref'])
    observation.update(active=True,observed_at_utc=guard._utc(),**facts)
    observation_ref=write_new('observation',observation)
    request={'key':key,'reservation_sha256':guard.budget.digest(own),
             'revocation_ref':context['permanent_revocation_ref'],'observation_ref':observation_ref}
    authority=guard.operation_draft_v1(guard.RETIRE_OPERATION,request,
             expected_before=context['expected_before'],bindings=context['bindings'])
    end=min(datetime.now(timezone.utc)+timedelta(minutes=10),datetime.fromisoformat(instruction['expires_at_utc']))
    authority.update(active=True,expires_at_utc=end.isoformat())
    result=write_new('authority',authority)
    finite()
    return result
