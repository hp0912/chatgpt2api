from __future__ import annotations

import json
import os
import unittest
from unittest import mock

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "test-auth")

from services import openai_backend_api
from services.config import config
from services.openai_backend_api import CODEX_RESPONSES_MODEL, ChatRequirements, OpenAIBackendAPI
from services.protocol import conversation, openai_v1_response
from utils.helper import is_image_chat_request


IMAGE_25_MODELS = ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst")


class ImageModelRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        for patcher in (
            mock.patch.object(openai_backend_api, "logger"),
            mock.patch.object(conversation, "logger"),
            mock.patch.object(conversation, "save_image_bytes", return_value="/images/test.png"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_web_prepare_and_generate_preserve_image_25_model_with_reference_images(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.com"
        backend.session = mock.Mock()
        response = backend.session.post.return_value
        response.status_code = 200
        response.json.return_value = {"conduit_token": "conduit-token"}
        reference = {
            "file_id": "file-reference", "file_name": "reference.png", "file_size": 10,
            "mime_type": "image/png", "width": 1024, "height": 1024,
        }
        for model in IMAGE_25_MODELS:
            for references in ([], [reference]):
                with (
                    self.subTest(model=model, edit=bool(references)),
                    mock.patch.object(backend, "_image_headers", return_value={}),
                    mock.patch.dict(config.data, {"default_upstream_model_name": "gpt-5-5", "default_thinking_effort": ""}),
                ):
                    backend.session.post.reset_mock()
                    token = backend._prepare_image_conversation("draw a cat", ChatRequirements("token"), model)
                    backend._start_image_generation("draw a cat", ChatRequirements("token"), token, model, references)

                self.assertEqual(token, "conduit-token")
                prepare, generate = backend.session.post.call_args_list
                self.assertTrue(prepare.args[0].endswith("/conversation/prepare"))
                self.assertEqual(prepare.kwargs["json"]["model"], model)
                payload = generate.kwargs["json"]
                self.assertEqual(payload["model"], model)
                self.assertEqual(payload["system_hints"], ["picture_v2"])
                self.assertEqual(len(payload["messages"][0]["content"]["parts"]), 1 + len(references))

    def test_legacy_web_model_still_uses_configured_upstream_and_thinking_effort(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        with mock.patch.dict(config.data, {"default_upstream_model_name": "gpt-5-5-extended"}):
            self.assertEqual(backend._image_model_settings("gpt-image-2"), ("gpt-5-5", "extended"))

    def test_codex_requests_select_the_requested_image_model_for_generate_and_edit(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        backend.access_token = "test-token"
        backend.base_url = "https://chatgpt.com"
        event = {"type": "image_generation_call", "result": "aW1hZ2U="}
        raw = mock.MagicMock()
        raw.__enter__.return_value = raw
        raw.headers = {"content-type": "text/event-stream"}
        raw.status = 200
        raw.read.return_value = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()

        for model in ("gpt-image-2", *IMAGE_25_MODELS):
            for prefix in ("codex-", "plus-codex-", "team-codex-", "pro-codex-"):
                for images in ([], ["data:image/png;base64,aW1hZ2U="]):
                    with (
                        self.subTest(model=prefix + model, edit=bool(images)),
                        mock.patch.object(openai_backend_api.account_service, "get_account", return_value={
                            "source_type": "codex", "type": "Plus",
                        }),
                        mock.patch.object(openai_backend_api.account_service, "_decode_jwt_payload", return_value={}),
                        mock.patch.object(openai_backend_api.urllib.request, "urlopen", return_value=raw) as send,
                    ):
                        events = list(backend.iter_codex_image_response_events(
                            prompt="draw a cat", model=prefix + model, images=images,
                            size="1536x1024", quality="high",
                        ))

                    payload = json.loads(send.call_args.args[0].data)
                    self.assertEqual(payload["model"], CODEX_RESPONSES_MODEL)
                    self.assertEqual(payload["tools"][0]["model"], model)
                    self.assertEqual(payload["tools"][0]["action"], "edit" if images else "generate")
                    self.assertEqual(payload["tools"][0]["size"], "1536x1024")
                    self.assertEqual(payload["tools"][0]["quality"], "high")
                    self.assertEqual(len(payload["input"][0]["content"]), 1 + len(images))
                    self.assertEqual(events, [event])

    def test_image_pool_routes_web_and_codex_models_to_the_correct_accounts(self) -> None:
        for model in ("gpt-image-2", *IMAGE_25_MODELS):
            for prefix, plan in (("", None), ("codex-", None), ("team-codex-", "team")):
                requested_model = prefix + model
                request = conversation.ConversationRequest(model=requested_model, prompt="draw a cat")
                output = conversation.ImageOutput(
                    kind="result", model=requested_model, index=1, total=1, data=[{"b64_json": "aW1hZ2U="}],
                )
                backend = mock.MagicMock()
                backend.iter_codex_image_response_events.return_value = iter([
                    {"type": "image_generation_call", "result": "aW1hZ2U="},
                ])
                with (
                    self.subTest(model=requested_model),
                    mock.patch.object(conversation, "OpenAIBackendAPI", return_value=backend),
                    mock.patch.object(conversation, "stream_image_outputs", return_value=iter([output])) as web,
                    mock.patch.object(conversation.account_service, "get_available_access_token", return_value="token") as select,
                    mock.patch.object(conversation.account_service, "get_account", return_value={}),
                    mock.patch.object(conversation.account_service, "mark_image_result"),
                ):
                    results = list(conversation.stream_image_outputs_with_pool(request))

                select.assert_called_once_with(
                    plan_type=plan, source_type="codex" if prefix else None,
                    plan_types=("plus", "team", "pro") if prefix and plan is None else None,
                )
                self.assertEqual(results[0].model, requested_model)
                self.assertTrue(is_image_chat_request({"model": requested_model}))
                if prefix:
                    web.assert_not_called()
                    self.assertEqual(backend.iter_codex_image_response_events.call_args.kwargs["model"], requested_model)
                else:
                    web.assert_called_once_with(backend, request, 1, 1)
                    backend.iter_codex_image_response_events.assert_not_called()
                backend.close.assert_called_once()

    def test_responses_uses_tool_model_and_preserves_top_level_model_in_both_modes(self) -> None:
        for model in IMAGE_25_MODELS:
            for prefix in ("", "codex-"):
                for stream in (False, True):
                    requested_model = prefix + model
                    output = conversation.ImageOutput(
                        kind="result", model=requested_model, index=1, total=1, data=[{"b64_json": "aW1hZ2U="}],
                    )
                    with (
                        self.subTest(model=requested_model, stream=stream),
                        mock.patch.object(openai_v1_response, "text_backend") as text_backend,
                        mock.patch.object(openai_v1_response, "count_text_tokens", return_value=3),
                        mock.patch.object(openai_v1_response, "stream_image_outputs_with_pool", return_value=iter([output])) as generate,
                    ):
                        result = openai_v1_response.handle({
                            "model": "gpt-5", "input": "draw a cat", "stream": stream,
                            "tools": [{"type": "image_generation", "model": requested_model, "quality": "high"}],
                        })
                        if stream:
                            result = list(result)[-1]["response"]

                    request = generate.call_args.args[0]
                    self.assertEqual(request.model, requested_model)
                    self.assertEqual(request.quality, "high")
                    self.assertEqual(result["model"], "gpt-5")
                    self.assertEqual(result["output"][0]["result"], "aW1hZ2U=")
                    text_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
