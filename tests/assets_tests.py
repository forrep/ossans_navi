import textwrap

from ossans_navi import assets


def test_get_information_obtained_by_rag_prompt():
    assert textwrap.dedent("""
    # Information obtained at RAG (JSON format)
    No relevant information was found. Please respond appropriately.
    """).strip() + "\n" == assets.get_information_obtained_by_rag_prompt([])

    assert textwrap.dedent("""
    # Information obtained at RAG (JSON format)
    ```
    [{\"timestamp\":\"2020-09-10 10:27:45\",\"name\":\"羽山 純\",\"user_id\":\"<@U7CAL37D0>\",\"content\":\"TS-UC会議で新PCについて意見をくださいということだったので送りますー\\n\\nXPS13 9300\\n- 総じてハードウェアには特に問題なしで、むしろかなりよい感じ。ただし利用までにはいくつかの設定を要した\\n- Audioドライバが正常に動作しなかった、マイクデバイスが認識されていない。関連アプリをアンインストールしてからドライバの再インストールで復帰\\n- Support Assistアプリが正常動作していなかった、ハードウェアのスキャンが途中で止まる、定期的に自動起動するものの毎回起動時にJSエラーが出るなど。アンインストール＆再インストールで復帰\\n- BIOS設定でFnロック状態で渡した方がいいのかも。みんなそれぞれ調べてBIOS設定を変えてそう\\n- （これは意見として）アプリ・ドライバの不具合はおそらくWindows10の2004の影響？まだ広くアップデートが配布されているステージじゃないので配布時点で2004にアップデートしなくてもいいんじゃないかと感じました\\n- グラフィック性能がわりと高いのが助かる。Iris Plusなのがよさそう。少し前に入れ替えたThinkPad勢はDiscordでカメラを起動するとカクカクしたり、GPUがすぐに100%になるらしい。今のWeb会議時代はGPUが結構重要\\n- カメラが旧PCに比べてキレイなのも良い感じ。暗くてもまあまあキレイに写る。採用面接で印象が悪いから外付け買おうかと思ってたけど、これなら不要そう。\\n- 4Kディスプレイ良い感じ、かなり描画がキレイ。ただし初期状態は色味がかなり暖色系。白が黄色に見える。特にデザ部とかはこのままじゃ厳しそう。\\n  - <https://www.dell.com/support/manuals/jp/ja/jpbsd1/xps-13-9300-laptop/xps-13-9300-setup-\",\"permalink\":\"https://slack.com/archives/C7FT308G1/p1599701265212100\",\"channel\":\"ts_kanri_ch\"},{\"timestamp\":\"2018-12-26 10:36:45\",\"name\":\"増井 淳之\",\"user_id\":\"<@U7G0W30PN>\",\"content\":\"搭載されている RAM が 8 GB 未満の一部の Lenovo ラップトップでは、KB4467691 をインストールした後、Windows が起動に失敗することがあります。\",\"permalink\":\"https://slack.com/archives/C7FT308G1/p1545788205008300\",\"channel\":\"ts_kanri_ch\"}]
    ```
    """).strip() + "\n" == assets.get_information_obtained_by_rag_prompt(
        [
            {
                'timestamp': '2020-09-10 10:27:45',
                'name': '羽山 純',
                'user_id': '<@U7CAL37D0>',
                'content': 'TS-UC会議で新PCについて意見をくださいということだったので送りますー\n\nXPS13 9300\n- 総じてハードウェアには特に問題なしで、むしろかなりよい感じ。ただし利用までにはいくつかの設定を要した\n- Audioドライバが正常に動作しなかった、マイクデバイスが認識されていない。関連アプリをアンインストールしてからドライバの再インストールで復帰\n- Support Assistアプリが正常動作していなかった、ハードウェアのスキャンが途中で止まる、定期的に自動起動するものの毎回起動時にJSエラーが出るなど。アンインストール＆再インストールで復帰\n- BIOS設定でFnロック状態で渡した方がいいのかも。みんなそれぞれ調べてBIOS設定を変えてそう\n- （これは意見として）アプリ・ドライバの不具合はおそらくWindows10の2004の影響？まだ広くアップデートが配布されているステージじゃないので配布時点で2004にアップデートしなくてもいいんじゃないかと感じました\n- グラフィック性能がわりと高いのが助かる。Iris Plusなのがよさそう。少し前に入れ替えたThinkPad勢はDiscordでカメラを起動するとカクカクしたり、GPUがすぐに100%になるらしい。今のWeb会議時代はGPUが結構重要\n- カメラが旧PCに比べてキレイなのも良い感じ。暗くてもまあまあキレイに写る。採用面接で印象が悪いから外付け買おうかと思ってたけど、これなら不要そう。\n- 4Kディスプレイ良い感じ、かなり描画がキレイ。ただし初期状態は色味がかなり暖色系。白が黄色に見える。特にデザ部とかはこのままじゃ厳しそう。\n  - <https://www.dell.com/support/manuals/jp/ja/jpbsd1/xps-13-9300-laptop/xps-13-9300-setup-',
                'permalink': 'https://slack.com/archives/C7FT308G1/p1599701265212100',
                'channel': 'ts_kanri_ch'
            },
            {
                'timestamp': '2018-12-26 10:36:45',
                'name': '増井 淳之',
                'user_id': '<@U7G0W30PN>',
                'content': '搭載されている RAM が 8 GB 未満の一部の Lenovo ラップトップでは、KB4467691 をインストールした後、Windows が起動に失敗することがあります。',
                'permalink': 'https://slack.com/archives/C7FT308G1/p1545788205008300',
                'channel': 'ts_kanri_ch'
            }
        ]
    )
