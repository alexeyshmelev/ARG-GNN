"""Authenticated short development registration. Import performs no file I/O."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LENGTH = 10_000_000
PLAN_REF = {
    'path': 'research/efficiency_20261027/integration_candidate/european_short_development_val/plan_v1.json',
    'sha256': '504cb68a9aaddfd9d1ba300b129b9101e92d4a87efccf92b93061b8e6c62f909',
}
SPEC_SHA = '816e2f5f9aa6d308c759a37a76d1e0cc3a1674dd38b8195214c0f7d9ee949f94'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def authenticate(ref):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'Exact source reference required')
    relative = Path(ref['path'])
    need(not relative.is_absolute() and '..' not in relative.parts, 'Repository-relative source required')
    path = ROOT / relative
    need(path.resolve() == path and path.is_file() and not path.is_symlink(), 'Canonical regular source required')
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Pinned source hash changed: ' + ref['path'])
    return raw


def read_json(ref):
    return json.loads(authenticate(ref))


def plan_ref():
    authenticate(PLAN_REF)
    return dict(PLAN_REF)


def validate_plan():
    plan = read_json(PLAN_REF)
    need(plan['specification_ref']['sha256'] == SPEC_SHA, 'Frozen independent specification required')
    specification = read_json(plan['specification_ref'])
    need(plan['source_refs'] == specification['source_refs'] and plan['storage'] == specification['storage'],
         'Exact source and aggregate resource contract required')
    for ref in plan['source_refs']:
        authenticate(ref)
    registration = read_json(plan['registration_ref'])
    need(canonical(registration) == canonical(specification['frozen_registration']), 'Frozen registration changed')
    recipe = read_json(plan['recipe_ref'])
    expected = read_json(specification['recipe_contract']['canonical_ref'])
    need(expected['parameters']['length_bp'] == 200_000_000, 'Canonical source length changed')
    expected['parameters']['length_bp'] = LENGTH
    need(canonical(recipe) == canonical(expected), 'Only the registered 200 Mb to 10 Mb length delta is allowed')
    for ref in recipe['source_refs']:
        authenticate(ref)
    seeds = registration['seeds']
    for key, preimage in registration['seed_preimages'].items():
        modulus = 2**31 - 7 if key.startswith('relate_') else 2**32 - 1
        value = 1 + int.from_bytes(hashlib.sha256(preimage.encode()).digest()[:8], 'big') % modulus
        need(value == seeds[key], 'Seed derivation changed')
    sampled = 1 + int.from_bytes(hashlib.sha256(registration['sampled_registry_seed_preimage'].encode()).digest()[:8], 'big') % (2**32 - 1)
    need(sampled == registration['sampled_registry_seed'], 'Sampled seed derivation changed')
    original = read_json(registration['original_eight_parent_registry_ref'])['parents']
    prior = read_json(registration['prior_prospective_200MbVAL_plan_ref'])['parents']
    need(len(original) == 8 and sum(p['split'] == 'train' for p in original) == 3
         and sum(p['split'] == 'test' for p in original) == 5 and len(prior) == 1,
         'Original three TRAIN, five TEST and prospective VAL registry required')
    def expanded(seed_map):
        return set(seed_map.values()) | {seed_map['relate_size_base'] + i for i in range(7)}
    historical = set().union(*(expanded(parent['seeds']) for parent in original + prior))
    fresh = expanded(seeds)
    need(len(fresh) == 10 and not fresh & historical and sampled not in historical | fresh,
         'Fresh streams including native size offsets 0 through 6 must be distinct')
    need(registration['parent_id'] not in {p['parent_id'] for p in original + prior}, 'Fresh parent identity required')
    sampling = read_json(plan['sampling_plan_ref'])
    need(sampling['sequence_length_bp'] == LENGTH and sampling['seed'] == sampled
         and list(sampling['parents']) == [registration['parent_id']], 'Short sampling plan binding changed')
    return dict(plan=plan, recipe=recipe, registration=registration, sampling_plan=sampling)


def validate_recipe(recipe):
    expected = validate_plan()['recipe']
    need(canonical(recipe) == canonical(expected), 'Exact frozen short recipe required')
    return recipe


def frozen_parent():
    registration = validate_plan()['registration']
    return {key: copy.deepcopy(registration[key]) for key in ('parent_id', 'split', 'seeds')}


def validate_parent(parent):
    need(canonical(parent) == canonical(frozen_parent()), 'Exact frozen short parent and four seed streams required')
    return parent


def validate_sampling_plan(plan):
    need(canonical(plan) == canonical(validate_plan()['sampling_plan']), 'Exact frozen short sampled plan required')
    return plan


def validate_native_config(config):
    plan = validate_plan()['plan']
    ref = next(ref for ref in plan['source_refs'] if ref['path'] == 'configs/benchmarks/ap200mb_native_runtime_v2.json')
    need(canonical(config) == canonical(read_json(ref)), 'Exact original native runtime settings required')
    return config
