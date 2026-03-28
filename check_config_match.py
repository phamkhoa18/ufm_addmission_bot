"""
KIEM TRA TOAN DIEN: Config YAML <-> Python Loader <-> Agent Node
Chay: python check_config_match.py
"""
import sys, os
sys.path.insert(0, ".")

passed = 0
failed = 0
warnings = 0

def ok(msg):
    global passed
    passed += 1
    print(f"  [OK] {msg}")

def fail(msg):
    global failed
    failed += 1
    print(f"  [FAIL] {msg}")

def warn(msg):
    global warnings
    warnings += 1
    print(f"  [WARN] {msg}")

# ============================================================
print("=" * 60)
print("PHAN 1: LOAD CONFIG YAML")
print("=" * 60)

try:
    from app.core.config import query_flow_config as cfg, models_yaml_data, prompts_yaml_data
    ok("query_flow_config loaded")
except Exception as e:
    fail(f"query_flow_config: {e}")
    sys.exit(1)

# ---- models_config.yaml keys ----
# NOTE: input_validation + keyword_filter KHONG co trong models_config (phi-model, nam trong guardian_config)
#       form key la "form" (chua ca extractor + drafter con)
expected_model_keys = [
    "long_query_summarizer", "prompt_guard_fast",
    "query_reformulation", "auto_summarize",
    "prompt_guard_deep", "multi_query", "embedding",
    "vector_router", "semantic_router", "main_bot",
    "care", "form",                               # form chua ca extractor + drafter
    "pr_query", "ufm_query", "web_search",
    "info_synthesizer", "pr_synthesizer", "sanitizer",
    "context_evaluator", "search_cache", "fallback_settings",
]
for k in expected_model_keys:
    if k in models_yaml_data:
        ok(f"models_config -> '{k}'")
    else:
        fail(f"models_config -> '{k}' MISSING")

# ---- prompts_config.yaml keys ----
expected_prompt_keys = [
    "input_validation", "long_query_summarizer", "keyword_filter",
    "prompt_guard_fast", "context_node", "auto_summarize",
    "prompt_guard_deep", "multi_query_node", "intent_classification",
    "pr_query_node", "ufm_query_node", "web_search_node",
    "info_synthesizer", "pr_synthesizer", "sanitizer_node",
    "context_evaluator", "form_extractor", "form_drafter",
    "main_bot", "fallback_messages", "response_templates",
]
for k in expected_prompt_keys:
    if k in prompts_yaml_data:
        ok(f"prompts_config -> '{k}'")
    else:
        fail(f"prompts_config -> '{k}' MISSING")

# ============================================================
print("\n" + "=" * 60)
print("PHAN 2: CONFIG LOADERS (Pydantic)")
print("=" * 60)

# Guardian loaders (tu guardian_config.yaml)
tests = [
    ("input_validation.max_input_chars", cfg.input_validation.max_input_chars, 3000),
    ("input_validation.summarize_threshold", cfg.input_validation.summarize_threshold, 1500),
    ("keyword_filter.banned_regex count", len(cfg.keyword_filter.banned_regex_patterns), 7),
    ("keyword_filter.teencode_map count", len(cfg.keyword_filter.teencode_map), 14),
]
for name, actual, expected in tests:
    if actual == expected:
        ok(f"{name} = {actual}")
    else:
        fail(f"{name}: expected={expected}, got={actual}")

# Model loaders (tu models_config.yaml)
model_tests = [
    ("long_query_summarizer.model", cfg.long_query_summarizer.model, "google/gemini-2.5-flash-lite"),
    ("prompt_guard_fast.model", cfg.prompt_guard_fast.model, "meta-llama/llama-prompt-guard-2-86m"),
    ("prompt_guard_deep.model", cfg.prompt_guard_deep.model, "qwen/qwen-2.5-7b-instruct"),
    ("query_reformulation.model", cfg.query_reformulation.model, "google/gemini-2.0-flash-001"),
    ("multi_query.model", cfg.multi_query.model, "google/gemini-2.0-flash-001"),
    ("embedding.model", cfg.embedding.model, "baai/bge-m3"),
    ("embedding.dimensions", cfg.embedding.dimensions, 1024),
    ("main_bot.model", cfg.main_bot.model, "google/gemini-2.0-flash-001"),
    ("main_bot.enabled", cfg.main_bot.enabled, True),
]
for name, actual, expected in model_tests:
    if actual == expected:
        ok(f"{name} = {actual}")
    else:
        fail(f"{name}: expected={expected}, got={actual}")

# Memory (tu query_context_config.yaml)
ok(f"memory.max_history_turns = {cfg.memory.max_history_turns}") if cfg.memory.max_history_turns == 10 else fail("memory.max_history_turns")

# Intent (tu intent_config.yaml + models_config.yaml)
ok(f"semantic_router.allowed_intents = {len(cfg.semantic_router.allowed_intents)}") if len(cfg.semantic_router.allowed_intents) == 17 else fail("allowed_intents count")
ok(f"intent_validator = {cfg.intent_validator.fallback_intent}") if cfg.intent_validator.fallback_intent == "KHONG_XAC_DINH" else fail("intent_validator")
ok(f"intent_threshold = {cfg.intent_threshold.min_query_length}") if cfg.intent_threshold.min_query_length == 5 else fail("intent_threshold")

# Intent routing
action = cfg.intent_actions.get_action("THONG_TIN_TUYEN_SINH")
ok(f"intent_actions('THONG_TIN_TUYEN_SINH') = {action}") if action == "PROCEED_RAG_UFM_SEARCH" else fail(f"intent_actions: {action}")
action2 = cfg.intent_actions.get_action("CHAO_HOI")
ok(f"intent_actions('CHAO_HOI') = {action2}") if action2 == "GREET" else fail(f"intent_actions: {action2}")
action3 = cfg.intent_actions.get_action("HO_TRO_SINH_VIEN")
ok(f"intent_actions('HO_TRO_SINH_VIEN') = {action3}") if action3 == "PROCEED_CARE" else fail(f"intent_actions: {action3}")

# RAG Search Pipeline
rag_tests = [
    ("pr_query.model", cfg.pr_query.model),
    ("ufm_query.model", cfg.ufm_query.model),
    ("web_search.model", cfg.web_search.model),
    ("info_synthesizer.model", cfg.info_synthesizer.model),
    ("pr_synthesizer.model", cfg.pr_synthesizer.model),
    ("sanitizer.model", cfg.sanitizer.model),
    ("context_evaluator.model", cfg.context_evaluator.model),
]
for name, val in rag_tests:
    if val:
        ok(f"{name} = {val}")
    else:
        fail(f"{name}: EMPTY")

# Fallback models chain
fm = cfg.fallback_models
for group, min_count in [("light", 2), ("medium", 2), ("search", 3), ("main", 3)]:
    chain = fm.get_model_chain(group)
    models = [m.model for m in chain]
    if len(chain) >= min_count:
        ok(f"fallback '{group}': {models}")
    else:
        fail(f"fallback '{group}': only {len(chain)} (need >= {min_count})")

# ============================================================
print("\n" + "=" * 60)
print("PHAN 3: PROMPT MANAGER")
print("=" * 60)

try:
    from app.core.prompts.manager import PromptManager
    pm = PromptManager()
    ok(f"PromptManager: {len(pm.list_domains())} domains")
except Exception as e:
    fail(f"PromptManager: {e}")
    pm = None

if pm:
    # System prompts
    sys_domains = [
        "context_node", "prompt_guard_deep", "multi_query_node",
        "intent_classification", "main_bot", "context_evaluator",
        "sanitizer_node", "info_synthesizer", "pr_synthesizer",
        "web_search_node", "pr_query_node", "ufm_query_node",
        "form_extractor", "form_drafter",
    ]
    for d in sys_domains:
        sp = pm.get_system(d)
        if sp and len(sp) > 10:
            ok(f"system('{d}'): {len(sp)} chars")
        else:
            fail(f"system('{d}'): EMPTY")

    # Compiled templates
    for d in ["context_node", "multi_query_node", "intent_classification",
              "main_bot", "context_evaluator", "sanitizer_node",
              "info_synthesizer", "pr_synthesizer", "pr_query_node",
              "ufm_query_node", "form_extractor", "form_drafter"]:
        if d in pm._compiled:
            ok(f"template('{d}'): compiled")
        else:
            warn(f"template('{d}'): NOT compiled")

    # Fallback messages
    for fk in ["cau_hoi_lac_de", "boi_nho_doi_thu", "doi_hoi_cam_ket",
                "tan_cong_he_thong", "out_of_scope"]:
        fb = pm.get_fallback(fk)
        if fb and len(fb) > 10:
            ok(f"fallback('{fk}')")
        else:
            fail(f"fallback('{fk}'): EMPTY")

    # Response templates
    for action in ["GREET", "CLARIFY"]:
        try:
            if action == "GREET":
                template = cfg.response_templates.greet_messages
            else:
                template = cfg.response_templates.clarify_messages
            
            if template and len(template) >= 1:
                ok(f"response_template('{action}'): OK")
            else:
                fail(f"response_template('{action}'): EMPTY")
        except AttributeError:
            fail(f"response_template('{action}'): AttributeError")

# ============================================================
print("\n" + "=" * 60)
print("PHAN 4: AGENT NODE IMPORTS")
print("=" * 60)

node_imports = [
    ("fast_scan_node", "app.services.langgraph.nodes.fast_scan_node"),
    ("context_node", "app.services.langgraph.nodes.context_node"),
    ("contextual_guard_node", "app.services.langgraph.nodes.contextual_guard_node"),
    ("multi_query_node", "app.services.langgraph.nodes.multi_query_node"),
    ("embedding_node", "app.services.langgraph.nodes.embedding_node"),
    ("intent_node", "app.services.langgraph.nodes.intent_node"),
    ("rag_node", "app.services.langgraph.nodes.rag_node"),
    ("care_node", "app.services.langgraph.nodes.care_node"),
    ("response_node", "app.services.langgraph.nodes.response_node"),
    ("pr_query_node", "app.services.langgraph.nodes.proceed_rag_search.pr_query_node"),
    ("web_search_node", "app.services.langgraph.nodes.proceed_rag_search.web_search_node"),
    ("synthesizer_node", "app.services.langgraph.nodes.proceed_rag_search.synthesizer_node"),
    ("sanitizer_node", "app.services.langgraph.nodes.proceed_rag_search.sanitizer_node"),
    ("evaluator", "app.services.langgraph.nodes.proceed_rag_search.evaluator"),
]
for name, module_path in node_imports:
    try:
        __import__(module_path)
        ok(f"import {name}")
    except Exception as e:
        fail(f"import {name}: {e}")

# ============================================================
print("\n" + "=" * 60)
print("PHAN 5: UTILS IMPORTS")
print("=" * 60)

for name, mod in [
    ("guardian_utils", "app.utils.guardian_utils"),
    ("query_summarizer", "app.utils.query_summarizer"),
    ("intent_service", "app.services.intent_service"),
    ("query_analyzer", "app.utils.query_analyzer"),
]:
    try:
        __import__(mod)
        ok(f"import {name}")
    except Exception as e:
        fail(f"import {name}: {e}")

# ============================================================
print("\n" + "=" * 60)
print("PHAN 6: CARE + FORM CONFIG")
print("=" * 60)

try:
    from app.core.config.care import CareConfig
    cc = CareConfig()
    c1 = cc.get_contact("ho_tro_sinh_vien")
    ok(f"care hotline: {c1.hotline}") if c1.hotline else fail("care: missing hotline")
    ok(f"care model: {cc.model}") if cc.model else fail("care model: EMPTY")
except Exception as e:
    fail(f"CareConfig: {e}")

try:
    from app.core.config.form_config import form_cfg
    ok(f"form extractor: {form_cfg.settings.extractor_model}") if form_cfg.settings.extractor_model else warn("form extractor: empty")
    ok(f"form drafter: {form_cfg.settings.drafter_model}") if form_cfg.settings.drafter_model else warn("form drafter: empty")
except Exception as e:
    fail(f"FormConfig: {e}")

# ============================================================
print("\n" + "=" * 60)
print(f"TONG KET: {passed} OK / {failed} FAIL / {warnings} WARN")
print("=" * 60)
if failed == 0:
    print(">>> HOAN HAO! Config YAML <-> Python <-> Agent Nodes da MATCH toan bo. <<<")
else:
    print(f">>> CON {failed} LOI CAN SUA! <<<")
