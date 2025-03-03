from ossans_navi.service.ai_service import AiResponse


def test_str_or_json_content():
    # 通常の文字列の場合はそのまま返す
    assert AiResponse.str_or_json_content("test string") == "test string"

    # JSONでcontentキーを持つ場合はその値を返す
    assert AiResponse.str_or_json_content('{"content": "test content"}') == "test content"

    # JSONだがcontentキーがない場合は元の文字列を返す
    assert AiResponse.str_or_json_content('{"key": "value"}') == '{"key": "value"}'

    # JSONでcontentの値が文字列でない場合は元の文字列を返す
    assert AiResponse.str_or_json_content('{"content": 123}') == '{"content": 123}'

    # 不正なJSONの場合は元の文字列を返す
    assert AiResponse.str_or_json_content('invalid json') == 'invalid json'
