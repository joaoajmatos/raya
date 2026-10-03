mod common;

use raya_tokenizer::JsonValue;

fn dumps(text: &str) -> String {
    JsonValue::parse(text).unwrap().dumps()
}

#[test]
fn keeps_key_order_and_python_separators() {
    assert_eq!(
        dumps(r#"{"task":"refund","customer":"ana","status":"late","amount":7}"#),
        r#"{"task": "refund", "customer": "ana", "status": "late", "amount": 7}"#
    );
    assert_eq!(
        dumps(r#"{"b":[1,2,{"z":null,"a":true}],"a":{}}"#),
        r#"{"b": [1, 2, {"z": null, "a": true}], "a": {}}"#
    );
    assert_eq!(dumps("[]"), "[]");
    assert_eq!(dumps("[ ]"), "[]");
}

#[test]
fn duplicate_keys_keep_first_position_and_last_value() {
    assert_eq!(dumps(r#"{"a":1,"b":2,"a":3}"#), r#"{"a": 3, "b": 2}"#);
}

#[test]
fn escapes_follow_ensure_ascii_false() {
    assert_eq!(
        dumps(r#""q\"b\\n\nt\tr\rb\bf\fu\u0001d\u007f é 日本 😀""#),
        "\"q\\\"b\\\\n\\nt\\tr\\rb\\bf\\fu\\u0001d\u{7f} é 日本 😀\""
    );
    assert_eq!(dumps(r#""\u001f""#), "\"\\u001f\"");
    // `/` is not escaped.
    assert_eq!(dumps(r#""a\/b""#), r#""a/b""#);
}

#[test]
fn numbers_match_python_repr() {
    for (input, want) in [
        ("7", "7"),
        ("-7", "-7"),
        ("18446744073709551615", "18446744073709551615"),
        ("7.0", "7.0"),
        ("0.5", "0.5"),
        ("-0.0", "-0.0"),
        ("0.0001", "0.0001"),
        ("0.00001", "1e-05"),
        ("1.5e-7", "1.5e-07"),
        ("123456789012345.0", "123456789012345.0"),
        ("1234567890123456.0", "1234567890123456.0"),
        ("1e16", "1e+16"),
        ("1.2345e20", "1.2345e+20"),
        ("1e300", "1e+300"),
        ("0.1", "0.1"),
        ("100.25", "100.25"),
        ("2.5e-5", "2.5e-05"),
    ] {
        assert_eq!(dumps(input), want, "{input}");
    }
}

#[test]
fn every_record_state_round_trips() {
    for name in common::record_names() {
        let rec = common::record(&name);
        let text = rec["state_serialized"].as_str().unwrap();
        if rec["state"].is_string() {
            continue;
        }
        assert_eq!(JsonValue::parse(text).unwrap().dumps(), text, "{name}");
    }
}
