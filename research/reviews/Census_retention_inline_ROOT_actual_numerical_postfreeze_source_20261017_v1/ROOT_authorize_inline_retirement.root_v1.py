"""Inactive numerical-only ROOT callback source. No work occurs at import."""
def authorize_inline_ROOT_retirement_v1(*,guard,context,finite):
    from datetime import datetime, timezone
    import hashlib, json, os, stat
    finite()
    def need(ok,message):
        if not ok:
            raise ValueError(message)
    directory='research/reviews/Census_retention_inline_ROOT_actual_numerical_postfreeze_source_20261017_v1'
    key='census_tsinfer_observation_v1/census-tsinfer-numerical-v1'
    paths={kind:directory+'/'+name for kind,name in (
        ('attempt','inline_attempt.root_v1.json'),
        ('observation','known_target_observation.root_active_v1.json'),
        ('authority','retirement_authority.root_active_v1.json'))}
    revocation_ref={'path':'research/reviews/Census_numerical_permanent_revocation_ROOT_20261017_v1/permanent_writer_revocation.root_active_v1.json',
        'sha256':'c7a4411960259dc536e25bbc2b0f19f4540301d9186413b0eb91b5a9e5cb019e'}
    selector_ref={'path':'research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json',
        'sha256':'fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4'}
    policy_ref={'path':'research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json',
        'sha256':'5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5'}
    need(set(context)=={'key','original_reservation','expected_before','bindings','root_instruction_ref',
        'fresh_metadata_paths','permanent_revocation_ref'},'Exact original callback context')
    need(str(guard.ROOT)=='/workspace-SR003.nfs2/ARG-GNN' and context['key']==key
        and context['fresh_metadata_paths']==paths and context['permanent_revocation_ref']==revocation_ref,
        'Exact numerical target, permanent revocation and immutable control names')
    need(guard._SELECTOR_REF==selector_ref and guard._POLICY_REF==policy_ref
        and guard.OPERATION_WALL_SECONDS==300 and guard.FRESH_TARGET_SECONDS==120,
        'Exact current Census and unchanged original deadlines')
    need(set(context['root_instruction_ref'])=={'path','sha256'}
        and context['root_instruction_ref']['path']==directory+'/ROOT_instruction.root_active_v1.json',
        'One explicit ROOT instruction; no prefix or path discovery')
    instruction=guard._json(context['root_instruction_ref'])
    need(instruction['active'] is True and instruction['root_authorizer']=='root_agent'
        and instruction['key']==key and instruction['metadata_directory']==directory
        and instruction['fresh_metadata_paths']==paths and instruction['permanent_revocation_ref']==revocation_ref
        and instruction['selector_ref']==selector_ref and instruction['policy_ref']==policy_ref
        and instruction['expected_before']==context['expected_before']
        and isinstance(context['expected_before'],dict) and isinstance(context['bindings'],dict)
        and instruction['expected_guard_roles']==context['bindings'] and len(context['bindings'])==47,
        'Keep ROOT explicit expected prefix and every47 role; never select another')
    guard._expire(instruction)
    finite()
    own=context['original_reservation']
    target=guard._target(key)
    need(own==target['original_reservation'] and own['key']==key
        and guard.budget.digest(own)=='e79a7c7b3806576efda20f606eb16fd0dbfd8f5fd4e7d14511cfe2a5785c16e5'
        and own['reserved_bytes']==20000000000 and own['auxiliary_scopes']==[]
        and own['output']==str(guard.ROOT)+'/experiments/results/'+key
        and target['original_expiry_unchanged']=='2026-10-11T11:40:36.315986+00:00'
        and target['known_PID_start_rows']==[
            {'pid':1306271,'role':'original_supervisor','start_ticks':436412146},
            {'pid':1306272,'role':'original_worker','start_ticks':436412154}]
        and target['known_process_groups']==[1306272],
        'Same original numerical reservation, sole scope, expiry and known identities')
    revocation=guard._revocation(revocation_ref,key)
    finite()
    facts=guard._observe_targets(target)
    observed=guard._utc()
    guard._validate_targets(facts,target,guard._stamp(revocation['revoked_at_utc']),
        guard._stamp(observed),fresh=True)
    finite()
    def write_new(kind,value):
        finite()
        path=guard.budget.v4.canonical(guard.ROOT,paths[kind])
        data=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode('utf-8')
        need(len(data)<=guard.MAX_JSON,'Original compact metadata byte cap')
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as stream:
            need(stream.write(data)==len(data),'Complete new immutable metadata write')
            stream.flush()
            os.fsync(stream.fileno())
            opened=os.fstat(stream.fileno())
        named=path.lstat()
        need(stat.S_ISREG(opened.st_mode) and stat.S_ISREG(named.st_mode)
            and opened.st_nlink==named.st_nlink==1 and opened.st_size==named.st_size==len(data)
            and (opened.st_dev,opened.st_ino)==(named.st_dev,named.st_ino),
            'New immutable ROOT metadata pathname changed')
        directory_fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        finite()
        return {'path':str(path.relative_to(guard.ROOT)),'sha256':hashlib.sha256(data).hexdigest()}
    observation=guard.stopped_observation_draft_v1(key,revocation_ref=revocation_ref)
    observation.update(active=True,observed_at_utc=observed,**facts)
    observation_ref=write_new('observation',observation)
    request={'key':key,'reservation_sha256':guard.budget.digest(own),
        'revocation_ref':revocation_ref,'observation_ref':observation_ref}
    authority=guard.operation_draft_v1(guard.RETIRE_OPERATION,request,
        expected_before=context['expected_before'],bindings=context['bindings'])
    guard._expire(instruction)
    authority.update(active=True,expires_at_utc=instruction['expires_at_utc'])
    result=write_new('authority',authority)
    finite()
    return result
