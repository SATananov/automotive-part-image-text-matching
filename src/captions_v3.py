from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartLanguage:
    noun: str
    subsystem: str
    function: str
    form: str


PART_LANGUAGE: dict[str, PartLanguage] = {
    "alternator": PartLanguage(
        noun="automotive alternator",
        subsystem="engine-support and electrical group",
        function="convert mechanical rotation into electrical power and recharge the battery",
        form="a compact ventilated metal housing with a pulley",
    ),
    "brake_disc": PartLanguage(
        noun="brake disc or rotor",
        subsystem="vehicle chassis group",
        function="provide the rotating friction surface clamped by the brake pads",
        form="a circular machined metal rotor, sometimes ventilated",
    ),
    "brake_pad": PartLanguage(
        noun="automotive brake pad",
        subsystem="vehicle chassis group",
        function="press friction material against a brake disc to slow the vehicle",
        form="a compact backing plate carrying a shaped friction lining",
    ),
    "coil_spring": PartLanguage(
        noun="suspension coil spring",
        subsystem="vehicle chassis group",
        function="support vehicle weight and store energy as the wheel moves",
        form="a helical steel spring with several visible coils",
    ),
    "headlight": PartLanguage(
        noun="front headlight assembly",
        subsystem="vehicle lighting group",
        function="illuminate the road ahead and make the vehicle visible from the front",
        form="a shaped lens and reflector assembly fitted to the front of a vehicle",
    ),
    "oil_filter": PartLanguage(
        noun="engine oil filter",
        subsystem="engine-support group",
        function="trap contaminants circulating in the engine lubricant",
        form="a replaceable canister or cartridge containing filtering media",
    ),
    "starter": PartLanguage(
        noun="automotive starter motor",
        subsystem="engine-support and electrical group",
        function="turn the engine crankshaft during starting using battery power",
        form="a compact electric motor with a solenoid and drive gear",
    ),
    "taillight": PartLanguage(
        noun="rear taillight assembly",
        subsystem="vehicle lighting group",
        function="show rear position, braking, and signalling information to other road users",
        form="a red or multicolour lens assembly fitted to the rear of a vehicle",
    ),
}

_TEMPLATE_BANKS: dict[str, tuple[str, ...]] = {
    "train": (
        "Automotive component: {noun}, part of the {subsystem}.",
        "Vehicle component identified as {noun}, designed to {function}.",
        "This component is the {noun}, recognizable by {form}.",
        "Road-vehicle replacement part: {noun}; its job is to {function}.",
    ),
    "validation": (
        "The pictured part is the {noun}; it helps {function}.",
        "This {subsystem} component is a {noun}, identifiable by {form}.",
    ),
}


def caption_count(split: str) -> int:
    templates = _TEMPLATE_BANKS.get(split)
    if templates is None:
        raise ValueError("Dataset V3 captions are defined only for train and validation.")
    return len(templates)


def render_caption(split: str, category: str, variant: int) -> str:
    if category not in PART_LANGUAGE:
        raise ValueError(f"Unknown Dataset V3 category: {category}")
    templates = _TEMPLATE_BANKS.get(split)
    if templates is None:
        raise ValueError("Dataset V3 captions are defined only for train and validation.")
    language = PART_LANGUAGE[category]
    return templates[variant % len(templates)].format(
        noun=language.noun,
        subsystem=language.subsystem,
        function=language.function,
        form=language.form,
    )


def all_captions(split: str) -> set[str]:
    return {
        render_caption(split, category, variant)
        for category in PART_LANGUAGE
        for variant in range(caption_count(split))
    }
