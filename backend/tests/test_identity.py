from __future__ import annotations

import unittest
from datetime import date
from typing import get_args

from backend.app.core.config import CLOUD_MODEL_PROVIDERS, Settings
from backend.app.core.identity import (
    ORION_CREATOR_NAME,
    _engine_for_provider,
    creator_age,
    creator_context,
    current_engine_fact,
    direct_creator_answer,
    institutional_identity_brief,
)
from backend.app.domain.schemas import StatusResponse


class OrionIdentityTests(unittest.TestCase):
    def test_creator_profile_is_available_as_validated_model_context(self) -> None:
        context = creator_context()
        self.assertIn(ORION_CREATOR_NAME, context)
        self.assertIn("16/01/2007", context)
        self.assertIn("Buenos Aires, Argentina", context)
        self.assertIn("Ciencia de Datos en ISTEA", context)
        self.assertIn("Atlético Mineiro", context)
        self.assertIn("Estudiantes de La Plata", context)
        self.assertIn("Excel", context)
        self.assertIn("Inteligencia Artificial", context)
        self.assertIn("PowerPoint", context)
        self.assertIn("VBA", context)
        self.assertIn("Licencia Pro", context)
        self.assertNotIn("Universidad de la Empresa", context)
        self.assertNotIn("Club Atlético Los Andes", context)

    def test_creator_age_is_derived_from_birth_date_instead_of_hardcoded(self) -> None:
        self.assertEqual(creator_age(date(2026, 1, 15)), 18)
        self.assertEqual(creator_age(date(2026, 1, 16)), 19)
        self.assertEqual(creator_age(date(2026, 8, 26)), 19)

    def test_no_creator_question_is_answered_by_python_phrase_matching(self) -> None:
        examples = (
            "¿Quién creó Orion?",
            "¿Quién te creó?",
            "Contame sobre la persona detrás de Orion",
            "¿Quién creó el motor de Orion?",
            "¿Quién creó el modelo gpt-oss que usa Orion?",
        )
        for query in examples:
            with self.subTest(query=query):
                self.assertIsNone(direct_creator_answer(query))

    def test_identity_context_distinguishes_orion_from_external_models(self) -> None:
        context = creator_context()
        self.assertIn("Santiago Vaccarini es el creador de Orion", context)
        self.assertIn("no creó gpt-oss", context)
        self.assertIn("Cloudflare Workers AI", context)
        self.assertIn("Ollama", context)
        self.assertIn("Interpretá semánticamente", context)

    def test_institutional_brief_resolves_default_referent_semantically(self) -> None:
        brief = institutional_identity_brief()
        self.assertIn(ORION_CREATOR_NAME, brief)
        self.assertIn("producto Orion", brief)
        self.assertIn("motor", brief)
        self.assertIn("no requiere búsqueda externa", brief)
        # Es contexto semántico, no una respuesta enlatada ni routing por frases.
        self.assertNotIn("¿", brief)

    def test_current_engine_fact_is_sourced_from_settings_not_hardcoded(self) -> None:
        # Live testing found Orion answering "GPT-4 by OpenAI" when asked which
        # engine powers it — a self-identity hallucination, since the real engine
        # is gpt-oss served via Cloudflare Workers AI. This fact must come from
        # settings (so it can't drift from the real deployment) and be stated
        # affirmatively so the model has a grounded answer instead of guessing.
        fact = current_engine_fact()
        self.assertIn("Cloudflare Workers AI", fact)
        self.assertIn("gpt-oss", fact)
        self.assertIn("No es GPT-4", fact)
        self.assertIn(fact, creator_context())
        self.assertIn(fact, institutional_identity_brief())

    def test_engine_fact_names_the_company_actually_serving_the_model(self) -> None:
        """The same weights come from several companies, so the host must follow config.

        Caught by running Orion on Cerebras: the fact was reading Cloudflare's
        settings unconditionally, so Orion would have stated a false thing about
        itself — the exact self-identity failure this fact exists to prevent.
        """

        cases = (
            (
                Settings(model_provider="cerebras", endpoint_quick_model="gpt-oss-120b"),
                "Cerebras",
                "gpt-oss-120b",
            ),
            (
                Settings(
                    model_provider="groq", endpoint_quick_model="openai/gpt-oss-120b"
                ),
                "Groq",
                "openai/gpt-oss-120b",
            ),
            (
                Settings(model_provider="ollama", quick_model="qwen3:4b-instruct"),
                "Ollama en esta computadora",
                "qwen3:4b-instruct",
            ),
            (
                Settings(model_provider="cloudflare"),
                "Cloudflare Workers AI",
                "@cf/openai/gpt-oss-120b",
            ),
        )
        for settings, expected_host, expected_model in cases:
            with self.subTest(provider=settings.model_provider):
                quick, _deep, host = _engine_for_provider(settings)
                self.assertEqual(host, expected_host)
                self.assertEqual(quick, expected_model)
                # No provider may leak another provider's name into the fact.
                for other in ("Cerebras", "Groq", "Cloudflare", "Ollama"):
                    if other not in expected_host:
                        self.assertNotIn(other, host)


class StatusSchemaTests(unittest.TestCase):
    def test_every_configurable_provider_is_accepted_by_the_status_route(self) -> None:
        """A provider missing from the schema breaks the whole status route.

        Caught live: selecting Cerebras made /status return a 500, because the
        response model still only allowed the two original providers.
        """

        declared = get_args(StatusResponse.model_fields["model_provider"].annotation)
        self.assertEqual(set(declared), CLOUD_MODEL_PROVIDERS | {"ollama"})


if __name__ == "__main__":
    unittest.main()
