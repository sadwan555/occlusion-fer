# Stage B Usage-routing amendment

Status: approved protocol change; implementation and HIVE revalidation required.

This amendment supersedes only the Stage 4 requirement that occlusion-enabled
routes reject a combined FER2013 CSV before opening. All other approved Stage 4
decisions remain unchanged.

Stage B now follows the locked E7 baseline loader contract:

- accept the official FER2013 combined CSV;
- inspect `Usage` first on every CSV row;
- parse `emotion` and `pixels` only for `Training` and `PublicTest` rows;
- preserve original CSV row numbers as sample IDs;
- compute Training and PublicTest canonical identities independently;
- exclude PrivateTest from typed records, hashes, Training mean, mask manifest,
  training, checkpoint selection, and PublicTest evaluation;
- retain permitted-splits JSON as a compatibility path for synthetic fixtures.

The source-routing identity is `combined-usage-routing-v1`. This change does not
alter `occlusion-v2-224`, mask geometry, condition order, ratios, seeds, the E7
recipe, checkpoint rule, or the PrivateTest final-evaluation gate.

Because a CSV parser must read row text to locate the `Usage` column, the
protocol does not claim that PrivateTest row bytes are never read. It claims the
narrow, testable property that excluded PrivateTest labels and pixels are not
parsed or consumed by any experiment artifact or model path.
