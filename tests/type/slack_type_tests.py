import datetime

from ossans_navi.type.slack_type import SlackSearchTerm


def test_slack_search_term():
    t1 = SlackSearchTerm(("ワード1", "ワード2", "ワード3",), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 8, 0, 0, 0, 0))
    t1_sub1 = SlackSearchTerm(("ワード1", "ワード2",), None, None)
    t1_sub2 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), None)
    t1_sub3 = SlackSearchTerm(("ワード1", "ワード2",), None, datetime.datetime(2024, 12, 8, 0, 0, 0, 0))
    t1_sub4 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 8, 0, 0, 0, 0))
    t1_sub5 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 7, 0, 0, 0, 0), datetime.datetime(2024, 12, 9, 0, 0, 0, 0))
    assert t1_sub1.is_subset(t1)
    assert t1_sub2.is_subset(t1)
    assert t1_sub3.is_subset(t1)
    assert t1_sub4.is_subset(t1)
    assert t1_sub5.is_subset(t1)

    t1_not_sub1 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), None)
    t1_not_sub2 = SlackSearchTerm(("ワード1", "ワード2",), None, datetime.datetime(2024, 12, 7, 0, 0, 0, 0))
    assert not t1_not_sub1.is_subset(t1)
    assert not t1_not_sub2.is_subset(t1)

    o1_1 = SlackSearchTerm(("ワード1", "ワード2",), None, None)
    o1_2 = SlackSearchTerm(("ワード1", "ワード2", "ワード3",), None, None)
    assert sorted([o1_1, o1_2]) == sorted([o1_2, o1_1]) == [o1_1, o1_2]

    o2_1 = SlackSearchTerm(("ワード1", "ワード2",), None, None)
    o2_2 = SlackSearchTerm(("ワード1", "ワーード2",), None, None)
    assert sorted([o2_1, o2_2]) == sorted([o2_2, o2_1]) == [o2_1, o2_2]

    o3_1 = SlackSearchTerm(("ワード1", "ワード2",), None, None)
    o3_2 = SlackSearchTerm(("ワード1", "ワーー2",), None, None)
    assert sorted([o3_1, o3_2]) == sorted([o3_2, o3_1]) == [o3_1, o3_2]

    o4_1 = SlackSearchTerm(("ワード1", "ワード2",), None, None)
    o4_2 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), None)
    o4_3 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), datetime.datetime(2024, 12, 7, 0, 0, 0, 0))
    assert sorted([o4_1, o4_2, o4_3]) == sorted([o4_3, o4_2, o4_1]) == [o4_1, o4_2, o4_3]

    o5_1 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 7, 0, 0, 0, 0), datetime.datetime(2024, 12, 5, 0, 0, 0, 0))
    o5_2 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 8, 0, 0, 0, 0), datetime.datetime(2024, 12, 6, 0, 0, 0, 0))
    o5_3 = SlackSearchTerm(("ワード1", "ワード2",), datetime.datetime(2024, 11, 9, 0, 0, 0, 0), datetime.datetime(2024, 12, 7, 0, 0, 0, 0))
    assert sorted([o5_1, o5_2, o5_3]) == sorted([o5_3, o5_2, o5_1]) == [o5_1, o5_2, o5_3]
