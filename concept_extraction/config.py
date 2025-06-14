#!/usr/bin/env python3
"""
Configuration file for Hybrid Concept Extraction System

This file contains all constants, patterns, and configurations used throughout
the concept extraction system.
"""

import re
from typing import Dict, Pattern

# ===================================================================
# REGEX PATTERNS FOR CONCEPT EXTRACTION
# ===================================================================

REGEX_MAP: Dict[str, Pattern[str]] = {
    # ───────────────────────────────────────────
    # 1. Gases and emissions
    # ───────────────────────────────────────────
    "co2": re.compile(r"\b(?:CO[₂2]|carbon\s+dioxide)(?:-eq)?\b", re.I),
    "ch4": re.compile(r"\b(?:CH4|methane)\b", re.I),
    "n2o": re.compile(r"\b(?:N2O|nitrous\s+oxide)\b", re.I),
    "ghg": re.compile(r"\b(?:GHGs?|greenhouse\s+gases?)\b", re.I),
    "f_gases": re.compile(r"\b(?:F-?gases|fluorinated\s+gases)\b", re.I),
    "co2_lulucf": re.compile(
        r"\b(?:CO\s*[₂2]\s*-\s*LULUCF|carbon\s+dioxide\s+from\s+LULUCF)\b", re.I
    ),
    "tco2_ffi": re.compile(
        r"\b(?:tCO[₂2]-FFI|tonnes\s+of\s+CO[₂2]\s+from\s+fossil\s+fuels\s+and\s+industry)\b",
        re.I,
    ),
    "pfcs": re.compile(r"\b(?:PFCs?|perfluorocarbons?)\b", re.I),
    "daccs": re.compile(
        r"\b(?:DACCS\+?|direct\s+air\s+carbon\s+capture\s+and\s+storage)\b", re.I
    ),
    "co2_eq": re.compile(r"\b(?:CO[₂2]-eq|carbon\s+dioxide\s+equivalen[ct]s?)\b", re.I),
    "hfcs": re.compile(r"\b(?:HFCs?|hydrofluorocarbons?)\b", re.I),
    "sf6": re.compile(
        r"\b(?:SF[₆6]|sulphur\s+hexafluoride|sulfur\s+hexafluoride)\b", re.I
    ),
    "o3": re.compile(r"\b(?:O[₃3]|ozone)\b", re.I),
    "emissions": re.compile(r"\bemissions?\b", re.I),
    "fossil fuels": re.compile(r"\bfossil\s+fuels?\b", re.I),
    "black carbon": re.compile(r"\bblack\s+carbon\b", re.I),
    "soot": re.compile(r"\bsoot\b", re.I),
    "non-co2 emissions": re.compile(r"\bnon[-\s]?co2\s+emissions?\b", re.I),
    "short-lived climate forcers": re.compile(
        r"\b(?:short[-\s]?lived\s+climate\s+forcers?|SLCFs?)\b", re.I
    ),
    "aerosols": re.compile(r"\baerosols?\b", re.I),
    "carbon dioxide removal": re.compile(
        r"\b(?:carbon\s+dioxide\s+removal|CDR)\b", re.I
    ),
    "beccs": re.compile(
        r"\b(?:BECCS|bioenergy\s+with\s+carbon\s+capture\s+and\s+storage)\b", re.I
    ),
    "ccs": re.compile(r"\b(?:CCS|carbon\s+capture\s+and\s+storage)\b", re.I),
    "ccu": re.compile(r"\b(?:CCU|carbon\s+capture\s+and\s+utili[sz]ation)\b", re.I),
    "cdr": re.compile(r"\b(?:CDR|carbon\s+dioxide\s+removal)\b", re.I),
    "co2_ffi": re.compile(
        r"\b(?:CO[₂2][-–]FFI|CO[₂2]\s+from\s+fossil\s+fuels\s+and\s+industry)\b", re.I
    ),
    "ffi": re.compile(r"\b(?:FFI|fossil\s+fuels?\s+and\s+industry)\b", re.I),
    "lulucf": re.compile(
        r"\b(?:LULUCF|land[-\s]use,\s+land[-\s]use\s+change\s+and\s+forestry)\b", re.I
    ),
    "nf3": re.compile(r"\b(?:NF3|nitrogen\s+trifluoride)\b", re.I),
    "rcb": re.compile(r"\bRCB\b", re.I),
    "rcp": re.compile(
        r"\b(?:RCPs?|representative\s+concentration\s+pathways?)\b", re.I
    ),
    "slcf": re.compile(r"\b(?:SLCFs?|short[-\s]?lived\s+climate\s+forcers?)\b", re.I),
    "tco2_eq": re.compile(
        r"\b(?:tCO[₂2]-eq|tonnes\s+of\s+CO[₂2]\s+equivalen[ct]s?)\b", re.I
    ),

    # ───────────────────────────────────────────
    # 2. Societal & economic terms
    # ───────────────────────────────────────────
    "ssp": re.compile(
        r"\b(?:SSP\d(?:-\d(?:\.\d)?)?|shared\s+socio[-\s]?economic\s+pathways?)\b", re.I
    ),
    "afolu": re.compile(
        r"\b(?:AFOLU|agriculture,\s*forestry?\s*and\s*other\s*land\s*use)\b", re.I
    ),
    "crd": re.compile(
        r"\bCRD\b", re.I
    ),
    "baseline": re.compile(r"\bbaseline\s+scenarios?\b", re.I),
    "mitigation": re.compile(r"\bmitigation\s+pathways?\b", re.I),
    "overshoot": re.compile(r"\bovershoot\s+scenarios?\b", re.I),
    "pathways": re.compile(r"\b1\.5\s*(?:°C|degrees\s*C)\s+pathways?\b", re.I),
    "low-emission": re.compile(r"\blow[-\s]?emission\s+scenarios?\b", re.I),
    "equality": re.compile(r"\bequality\b", re.I),
    "inequality": re.compile(r"\binequality\b", re.I),
    "equity": re.compile(r"\bequity\b", re.I),
    "security": re.compile(r"\bsecurity\b", re.I),
    "sustainability": re.compile(r"\bsustainabilit(?:y|ies)\b", re.I),
    "fao": re.compile(
        r"\b(?:FAO|food\s+and\s+agriculture\s+organisation|food\s+and\s+agriculture\s+organization)\b",
        re.I,
    ),
    "gdp": re.compile(r"\b(?:GDP|gross\s+domestic\s+product)\b", re.I),
    "iea_steps": re.compile(
        r"\b(?:IEA[-\s]?STEPS|international\s+energy\s+agency\s+stated\s+polic(?:y|ies)\s+scenario)\b",
        re.I,
    ),
    "imp_ren": re.compile(r"\bIMP\s*[-–]?\s*REN\b", re.I),
    "ldc": re.compile(r"\b(?:LDCs?|least\s+developed\s+countries?)\b", re.I),
    "lk": re.compile(
        r"\bLK\b", re.I
    ),
    "ndc": re.compile(r"\b(?:NDCs?|nationally\s+determined\s+contributions?)\b", re.I),
    "r_and_d": re.compile(r"\b(?:R\s*&?\s*D|research\s+and\s+development)\b", re.I),
    "sdg": re.compile(r"\b(?:SDGs?|sustainable\s+development\s+goals?)\b", re.I),
    "sdps": re.compile(r"\bSDPs?\b", re.I),
    "sids": re.compile(r"\b(?:SIDS|small\s+island\s+developing\s+states?)\b", re.I),
    "unfccc": re.compile(
        r"\b(?:UNFCCC|united\s+nations\s+framework\s+convention\s+on\s+climate\s+change)\b",
        re.I,
    ),
    "usd": re.compile(r"\b(?:USD|US\s*dollars?)\b", re.I),

    # ───────────────────────────────────────────
    # 3. Climate-change concepts
    # ───────────────────────────────────────────
    "net_zero": re.compile(r"\bnet\s+zero\b", re.I),
    "net_zero_ghg": re.compile(
        r"\bnet[-\s]?zero\s+(?:GHGs?|greenhouse\s+gases?|CO[₂2]|carbon\s+dioxide)\b",
        re.I,
    ),
    "carbon_budget": re.compile(r"\b(?:remaining\s+)?carbon\s+budget\b", re.I),
    "adaptation": re.compile(r"\badaptation(?:\s+(?:gap|limits?))?\b", re.I),
    "resilience": re.compile(r"\bresilien[ct]\w*\b", re.I),
    "imp_sp": re.compile(r"\bIMP\s*-\s*SP\b", re.I),
    "srm": re.compile(r"\b(?:SRM|solar\s+radiation\s+modification)\b", re.I),
    "drm": re.compile(r"\b(?:DRM|disaster\s+risk\s+management)\b", re.I),
    "eba": re.compile(r"\b(?:EbA|ecosystem[-\s]based\s+adaptation)\b", re.I),
    "gwl": re.compile(r"\b(?:GWL|global\s+warming\s+level(?:s)?)\b", re.I),
    "imp": re.compile(r"\b(?:IMP|(?i:illustrative\s+mitigation\s+pathways?))\b"),
    "imp_ld": re.compile(r"\bIMP\s*[-–]?\s*LD\b", re.I),
    "imp_neg": re.compile(r"\bIMP\s*[-–]?\s*NEG\b", re.I),
    "ip_modact": re.compile(r"\bIP[-–]?ModAct\b", re.I),
    "wim": re.compile(r"\b(?:WIM|warsaw\s+international\s+mechanism)\b", re.I),
    "abrupt_climate_change": re.compile(r"\babrupt\s+climate\s+change\b", re.I),
    "afforestation": re.compile(r"\bafforestat(?:ion|e|ed)\b", re.I),
    "bioenergy": re.compile(r"\bbioenergy\b", re.I),
    "cascading_impacts": re.compile(r"\bcascading\s+impacts?\b", re.I),
    "deforestation": re.compile(r"\bdeforestat(?:ion|e|ed)\b", re.I),
    "maladaptation": re.compile(r"\bmaladapt(?:ation|ive)\b", re.I),
    "residual_risk": re.compile(r"\bresidual\s+risk\b", re.I),
    "vulnerability": re.compile(r"\bvulnerab(?:le|ility)\b", re.I),

    # ───────────────────────────────────────────
    # 4. Temperatures & thresholds
    # ───────────────────────────────────────────
    "temp_1_5c": re.compile(r"\b1\.5\s*°?C\b", re.I),
    "temp_2c": re.compile(r"\b2(?:\.0)?\s*°?C\b", re.I),
    "temp_xc": re.compile(r"\b\d(?:\.\d)?\s*°?C\b"),
    "temperature anomaly": re.compile(r"\btemperature\s+anomal(?:y|ies)\b", re.I),
    "ecs": re.compile(r"\b(?:ECS|equilibrium\s+climate\s+sensitivity)\b", re.I),

    # ───────────────────────────────────────────
    # 5. Observed / projected changes
    # ───────────────────────────────────────────
    "sea_level_rise": re.compile(r"\bsea\s+level\s+rise\b", re.I),
    "global_temp": re.compile(
        r"\b(?:global\s+surface\s+temperature|warming\s+level(?:s)?)\b", re.I
    ),
    "ice_sheet": re.compile(r"\bice\s+sheet(?:s)?\b", re.I),
    "precipitation": re.compile(r"\bprecipitation\b", re.I),
    "weather": re.compile(r"\bweather\s+(?:patterns?|events?)\b", re.I),
    "climate": re.compile(r"\bclimate\s+(?:patterns?|systems?|variability)\b", re.I),
    "global": re.compile(r"\bglobal\s+(?:warming|temperature)\b", re.I),
    "temperature": re.compile(r"\btemperature\s+(?:rise|increase|change)\b", re.I),
    "radiative": re.compile(r"\bradiative\s+forcing\b", re.I),
    "cid": re.compile(r"\b(?:CID|climate\s+impact\s+drivers?)\b", re.I),
    "heatwaves": re.compile(r"\bheat\s*waves?\b", re.I),
    "extreme": re.compile(r"\bextreme\s+(?:weather|events?|temperature)\b", re.I),
    "gt": re.compile(r"\b(?:GT|gigatonnes?)\b", re.I),

    # ───────────────────────────────────────────
    # 6. Impact & Units
    # ───────────────────────────────────────────
    "gw": re.compile(r"\b(?:GW|gigawatts?)\b", re.I),
    "w_m2": re.compile(r"\bW\s*m[-²2]?\b", re.I),
    "gwp100": re.compile(r"\b(?:GWP100|100[-\s]?year\s+global\s+warming\s+potential)\b", re.I),
    "kwh": re.compile(r"\b(?:kWh|kilowatt[-\s]?hours?)\b", re.I),
    "lcoe": re.compile(r"\b(?:LCOE|levelized\s+cost\s+of\s+energy)\b", re.I),
    "mwh": re.compile(r"\b(?:MWh|megawatt[-\s]?hours?)\b", re.I),
    "ppp": re.compile(r"\b(?:PPP|purchasing\s+power\s+parity)\b", re.I),
    "gigatonnes": re.compile(r"\bgigatonnes?\b", re.I),
    "ppm": re.compile(r"\bppm\b", re.I),
    "ppb": re.compile(r"\bppb\b", re.I),

    # ───────────────────────────────────────────
    # 7. Reports & Assessment
    # ───────────────────────────────────────────
    "srocc": re.compile(
        r"\b(?:SROCC|special\s+report\s+on\s+(?:the\s+)?ocean\s+and\s+cryosphere)\b", re.I
    ),
    "ar5": re.compile(
        r"\b(?:AR5|fifth\s+assessment\s+report)\b", re.I
    ),
    "ar6": re.compile(r"\b(?:AR6|sixth\s+assessment\s+report)\b", re.I),
    "sr1_5": re.compile(
        r"\b(?:SR1\.5|SR15|special\s+report\s+on\s+global\s+warming\s+of\s+1\.5\s*°?C)\b",
        re.I,
    ),
    "srccl": re.compile(
        r"\b(?:SRCCL|special\s+report\s+on\s+climate\s+change\s+and\s+land)\b", re.I
    ),
    "syr": re.compile(r"\b(?:SYR|synthesis\s+report)\b", re.I),
    "csb": re.compile(r"\bCSB\b", re.I),
    "rfcs": re.compile(r"\b(?:RFCs?|reasons\s+for\s+concern)\b", re.I),

    # ───────────────────────────────────────────
    # 8. Models
    # ───────────────────────────────────────────
    "cmip5": re.compile(
        r"\b(?:CMIP5|coupled\s+model\s+intercomparison\s+project\s+phase\s*5)\b", re.I
    ),
    "cmip6": re.compile(
        r"\b(?:CMIP6|coupled\s+model\s+intercomparison\s+project\s+phase\s*6)\b", re.I
    ),
    "fair": re.compile(
        r"\b(?:FaIR|FAIR|(?i:finite[-\s]amplitude\s+impulse\s+response))\b"
    ),
    "magicc": re.compile(
        r"\b(?:MAGICC|model\s+for\s+the\s+assessment\s+of\s+greenhouse[-\s]gas\s+induced\s+climate\s+change)\b",
        re.I,
    ),

    # ───────────────────────────────────────────
    # 9. Technology
    # ───────────────────────────────────────────
    "ev": re.compile(r"\b(?:EVs?|electric\s+vehicles?)\b", re.I),
    "ews": re.compile(r"\b(?:EWS|early\s+warning\s+systems?)\b", re.I),
    "li_on": re.compile(r"\b(?:Li[-\s]?ion|lithium[-\s]?ion)\b", re.I),
    "pv": re.compile(r"\b(?:PV|photovoltaic(?:s)?)\b", re.I),
    
    # ───────────────────────────────────────────
    # 10. Agencies & organisations
    # ───────────────────────────────────────────
    "iea": re.compile(r"\b(?:IEA|international\s+energy\s+agency)\b", re.I),
    "ipcc": re.compile(
        r"\b(?:IPCC|intergovernmental\s+panel\s+on\s+climate\s+change)\b", re.I
    ),
    "who": re.compile(
        r"\b(?:WHO|(?i:world\s+health\s+(?:organisation|organization)))\b"
    ),
}

# ===================================================================
# PATTERN SCHEMA - MAPS REGEX KEYS TO CONCEPT CATEGORIES
# ===================================================================

PATTERN_SCHEMA: Dict[str, str] = {
    # EMISSION
    "co2": "EMISSION",
    "ch4": "EMISSION",
    "n2o": "EMISSION",
    "ghg": "EMISSION",
    "f_gases": "EMISSION",
    "gigatonnes": "EMISSION",
    "ppm": "EMISSION",
    "ppb": "EMISSION",
    "co2_lulucf": "EMISSION",
    "tco2_ffi": "EMISSION",
    "pfcs": "EMISSION",
    "daccs": "EMISSION",
    "co2_eq": "EMISSION",
    "hfcs": "EMISSION",
    "sf6": "EMISSION",
    "o3": "EMISSION",
    "emissions": "EMISSION",
    "fossil fuels": "EMISSION",
    "black carbon": "EMISSION",
    "soot": "EMISSION",
    "non-co2 emissions": "EMISSION",
    "short-lived climate forcers": "EMISSION",
    "aerosols": "EMISSION",
    "carbon dioxide removal": "EMISSION",
    "beccs": "EMISSION",
    "ccs": "EMISSION",
    "ccu": "EMISSION",
    "cdr": "EMISSION",
    "co2_ffi": "EMISSION",
    "ffi": "EMISSION",
    "lulucf": "EMISSION",
    "nf3": "EMISSION",
    "rcb": "EMISSION",
    "rcp": "EMISSION",
    "slcf": "EMISSION",
    "tco2_eq": "EMISSION",

    # SCENARIO
    "ssp": "SCENARIO",
    "rcp": "SCENARIO",
    "afolu": "SCENARIO",
    "crd": "SCENARIO",
    "baseline": "SCENARIO",
    "pathways": "SCENARIO",
    "low-emission": "SCENARIO",
    "fao": "SCENARIO",
    "imp_ren": "SCENARIO",
    "lk": "SCENARIO",
    "r_and_d": "SCENARIO",
    "sdg": "SCENARIO",
    "sids": "SCENARIO",

    # CLIMATE_VAR
    "global_temp": "CLIMATE_VAR",
    "sea_level_rise": "CLIMATE_VAR",
    "ice_sheet": "CLIMATE_VAR",
    "temp_1_5c": "CLIMATE_VAR",
    "temp_2c": "CLIMATE_VAR",
    "temp_xc": "CLIMATE_VAR",
    "srm": "CLIMATE_VAR",
    "cid": "CLIMATE_VAR",
    "precipitation": "CLIMATE_VAR",
    "weather": "CLIMATE_VAR",
    "climate": "CLIMATE_VAR",
    "global": "CLIMATE_VAR",
    "temperature": "CLIMATE_VAR",
    "radiative": "CLIMATE_VAR",
    "ecs": "CLIMATE_VAR",
    "gwl": "CLIMATE_VAR",
    "imp": "CLIMATE_VAR",
    "imp_ld": "CLIMATE_VAR",
    "imp_neg": "CLIMATE_VAR",
    "ip_modact": "CLIMATE_VAR",
    "heatwaves": "CLIMATE_VAR",
    "extreme": "CLIMATE_VAR",
    
    # POLICY
    "net_zero": "POLICY",
    "carbon_budget": "POLICY",
    "mitigation": "POLICY",
    "adaptation": "POLICY",
    "overshoot": "POLICY",
    "imp_sp": "POLICY",
    "equality": "POLICY",
    "inequality": "POLICY",
    "equity": "POLICY",
    "security": "POLICY",
    "sustainability": "POLICY",
    "drm": "POLICY",
    "eba": "POLICY",
    "gdp": "POLICY",
    "iea_steps": "POLICY",
    "ldc": "POLICY",
    "ndc": "POLICY",
    "sdps": "POLICY",
    "unfccc": "POLICY",
    "usd": "POLICY",
    "resilience": "POLICY",
    "wim": "POLICY",
    "abrupt_climate_change": "POLICY",
    "afforestation": "POLICY",
    "bioenergy": "POLICY",
    "cascading_impacts": "POLICY",
    "deforestation": "POLICY",
    "maladaptation": "POLICY",
    "residual_risk": "POLICY",
    "vulnerability": "POLICY",

    # IMPACT
    "usd": "IMPACT",
    "gw": "IMPACT",
    "w_m2": "IMPACT",
    "gwp100": "IMPACT",
    "kwh": "IMPACT",
    "lcoe": "IMPACT",
    "mwh": "IMPACT",
    "ppp": "IMPACT",
    "gt": "IMPACT",

    # REPORTS
    "srocc": "REPORTS",
    "ar5": "REPORTS",
    "ar6": "REPORTS",
    "csb": "REPORTS",
    "rfcs": "REPORTS",
    "sr1_5": "REPORTS",
    "srccl": "REPORTS",
    "syr": "REPORTS",

    # MODELS
    "cmip5": "MODELS",
    "cmip6": "MODELS",
    "fair": "MODELS",
    "magicc": "MODELS",

    # TECHNOLOGY
    "ev": "TECHNOLOGY",
    "ews": "TECHNOLOGY",
    "li_on": "TECHNOLOGY",
    "pv": "TECHNOLOGY",

    # AGENCIES
    "iea": "AGENCIES",
    "ipcc": "AGENCIES",
    "who": "AGENCIES",

    # TEMPORAL
    "temperature anomaly": "TEMPORAL",
}

# ===================================================================
# DOMAIN-SPECIFIC PATTERNS FOR ADVANCED CONCEPT EXTRACTION
# ===================================================================

DOMAIN_SPECIFIC_PATTERNS: Dict[str, list] = {
    'CLIMATE_PHENOMENON': [
        r'\b(?:el\s+ni[ñn]o|la\s+ni[ñn]a)\b',
        r'\b(?:arctic\s+oscillation|ao)\b',
        r'\b(?:north\s+atlantic\s+oscillation|nao)\b',
        r'\b(?:pacific\s+decadal\s+oscillation|pdo)\b',
        r'\b(?:atlantic\s+multidecadal\s+oscillation|amo)\b',
        r'\b(?:indian\s+ocean\s+dipole|iod)\b',
        r'\b(?:southern\s+annular\s+mode|sam)\b',
        r'\b(?:madden[-\s]?julian\s+oscillation|mjo)\b'
    ],
    'CLIMATE_SYSTEM': [
        r'\b(?:thermohaline\s+circulation|thc)\b',
        r'\b(?:meridional\s+overturning\s+circulation|moc)\b',
        r'\b(?:atlantic\s+meridional\s+overturning\s+circulation|amoc)\b',
        r'\b(?:polar\s+vortex)\b',
        r'\b(?:jet\s+stream)\b',
        r'\b(?:hadley\s+cell)\b',
        r'\b(?:walker\s+circulation)\b'
    ],
    'IMPACT_METRIC': [
        r'\b(?:degree\s+days?|heating\s+degree\s+days?|cooling\s+degree\s+days?)\b',
        r'\b(?:growing\s+season\s+length|gsl)\b',
        r'\b(?:frost\s+days?)\b',
        r'\b(?:heat\s+index)\b',
        r'\b(?:wet\s+bulb\s+temperature)\b',
        r'\b(?:vapor\s+pressure\s+deficit|vpd)\b'
    ]
}

# ===================================================================
# ENTITY RULER PATTERNS FOR HIGH-CONFIDENCE EXACT MATCHES
# ===================================================================

ENTITY_RULER_PATTERNS: Dict[str, list] = {
    "co2": ["CO2", "CO₂", "carbon dioxide"],
    "ch4": ["CH4", "methane"],
    "n2o": ["N2O", "nitrous oxide"],
    "ghg": ["GHG", "GHGs", "greenhouse gas", "greenhouse gases"],
    "net_zero": ["net zero", "net-zero"],
    "ipcc": ["IPCC", "Intergovernmental Panel on Climate Change"],
    "ar6": ["AR6", "Sixth Assessment Report"],
    "sr1_5": ["SR1.5", "Special Report on Global Warming of 1.5°C"],
    "temp_1_5c": ["1.5°C", "1.5 degrees Celsius"],
    "temp_2c": ["2°C", "2.0°C", "2 degrees Celsius"],
    "sea_level_rise": ["sea level rise", "sea-level rise"],
    "fossil_fuels": ["fossil fuels", "fossil fuel"],
    "renewable_energy": ["renewable energy", "clean energy"],
    "carbon_budget": ["carbon budget", "remaining carbon budget"],
    "paris_agreement": ["Paris Agreement", "Paris Climate Agreement"]
}

# ===================================================================
# NER MODEL CONFIGURATIONS
# ===================================================================

# Default spaCy model
SPACY_MODEL = "en_core_web_sm"

# Transformers model for NER (set to None to disable and use spaCy + NLTK only)
TRANSFORMERS_NER_MODEL = None  # "dbmdz/bert-large-cased-finetuned-conll03-english"

# NLTK stopwords language
NLTK_STOPWORDS_LANG = "english"

# ===================================================================
# LOGGING CONFIGURATION
# ===================================================================

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = "INFO"

# ===================================================================
# PROCESSING PARAMETERS
# ===================================================================

# Batch size for processing large datasets
BATCH_SIZE = 1000

# Progress report interval
PROGRESS_INTERVAL = 1000

# Confidence thresholds for different extraction methods
CONFIDENCE_THRESHOLDS = {
    'entity_ruler': 1.0,
    'regex': 1.0,
    'domain_specific': 0.9,
    'transformers': 0.8,
    'spacy': 0.7,
    'nltk': 0.6
}

# Priority order for concept merging (higher number = higher priority)
EXTRACTION_PRIORITY = {
    'entity_ruler': 6,
    'regex': 5,
    'domain_specific': 4,
    'transformers': 3,
    'spacy': 2,
    'nltk': 1
}