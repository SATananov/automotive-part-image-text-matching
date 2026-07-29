from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartLanguage:
    noun: str
    system: str
    function: str
    form: str


PART_LANGUAGE: dict[str, PartLanguage] = {
    "air_filter": PartLanguage(
        noun="engine air filter",
        system="air-intake and filtration system",
        function="remove dust and particles from the air entering the engine",
        form="pleated filtering media inside a rigid frame or housing",
    ),
    "alternator": PartLanguage(
        noun="automotive alternator",
        system="vehicle charging and electrical system",
        function="convert mechanical rotation into electrical power and recharge the battery",
        form="a compact metal housing with a pulley and ventilation openings",
    ),
    "brake_disc": PartLanguage(
        noun="brake disc or rotor",
        system="disc-brake system",
        function="provide the rotating friction surface clamped by the brake pads",
        form="a circular machined metal rotor, sometimes ventilated",
    ),
    "brake_pad": PartLanguage(
        noun="automotive brake pad",
        system="disc-brake system",
        function="press friction material against a brake disc to slow the vehicle",
        form="a compact backing plate carrying a shaped friction lining",
    ),
    "coil_spring": PartLanguage(
        noun="suspension coil spring",
        system="vehicle suspension system",
        function="support vehicle weight and store energy as the wheel moves",
        form="a helical steel spring with several visible coils",
    ),
    "headlight": PartLanguage(
        noun="front headlight assembly",
        system="vehicle lighting system",
        function="illuminate the road ahead and make the vehicle visible from the front",
        form="a shaped lens and reflector assembly for the front of a vehicle",
    ),
    "oil_filter": PartLanguage(
        noun="engine oil filter",
        system="engine lubrication and filtration system",
        function="trap contaminants circulating in the engine lubricant",
        form="a replaceable canister or cartridge containing filtering media",
    ),
    "shock_absorber": PartLanguage(
        noun="suspension shock absorber",
        system="vehicle suspension system",
        function="dampen spring movement and control wheel and body oscillation",
        form="a telescopic hydraulic damper with a cylindrical body and piston rod",
    ),
    "starter": PartLanguage(
        noun="automotive starter motor",
        system="engine starting and electrical system",
        function="turn the engine crankshaft during starting using battery power",
        form="a compact electric motor with a solenoid and drive gear",
    ),
    "taillight": PartLanguage(
        noun="rear taillight assembly",
        system="vehicle lighting and signalling system",
        function="show rear position, braking, and signalling information to other road users",
        form="a red or multi-colour lens assembly mounted at the rear of a vehicle",
    ),
}


_TEMPLATE_BANKS: dict[str, tuple[str, ...]] = {
    "train": (
        "An automotive {noun} used in the {system}.",
        "A vehicle {noun} designed to {function}.",
        "This component is a {noun} with {form}.",
        "A replacement {noun} for a road vehicle; its job is to {function}.",
    ),
    "validation": (
        "The pictured part is a {noun}; it helps {function}.",
        "A {noun} associated with the {system}, recognizable by {form}.",
    ),
    "test": (
        "A motor-vehicle {noun} whose role is to {function}.",
        "This {system} component is identifiable as a {noun} with {form}.",
    ),
}


def caption_count(split: str) -> int:
    if split not in _TEMPLATE_BANKS:
        raise ValueError(f"Unknown caption split: {split}")
    return len(_TEMPLATE_BANKS[split])


def render_caption(split: str, category: str, variant: int) -> str:
    if category not in PART_LANGUAGE:
        raise ValueError(f"Unknown part category: {category}")
    templates = _TEMPLATE_BANKS.get(split)
    if templates is None:
        raise ValueError(f"Unknown caption split: {split}")
    language = PART_LANGUAGE[category]
    template = templates[variant % len(templates)]
    return template.format(
        noun=language.noun,
        system=language.system,
        function=language.function,
        form=language.form,
    )


def all_captions(split: str) -> set[str]:
    return {
        render_caption(split, category, variant)
        for category in PART_LANGUAGE
        for variant in range(caption_count(split))
    }
