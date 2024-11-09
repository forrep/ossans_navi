import datetime

from ossans_navi.type.slack_type import SlackSearchTerm


def test_slack_search_term():
    assert (
        SlackSearchTerm(("ワード1", "ワード2",), None, None).is_subset(
            SlackSearchTerm(("ワード1", "ワード2", "ワード3"), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 8, 0, 0, 0, 0))
        )
    )

    assert not (
        SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), None).is_subset(
            SlackSearchTerm(("ワード1", "ワード2", "ワード3"), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 8, 0, 0, 0, 0))
        )
    )

    assert not (
        SlackSearchTerm(("ワード1", "ワード2",), None, datetime.datetime(2024, 12, 7, 0, 0, 0, 0)).is_subset(
            SlackSearchTerm(("ワード1", "ワード2", "ワード3"), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 8, 0, 0, 0, 0))
        )
    )

    assert (
        SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 8, 0, 0, 0, 0)).is_subset(
            SlackSearchTerm(("ワード1", "ワード2", "ワード3"), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 8, 0, 0, 0, 0))
        )
    )
