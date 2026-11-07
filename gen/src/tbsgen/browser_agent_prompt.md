# tbsgen S3 browser batch — agent instructions

You download the full text (PDF) of a few research papers through the user's browser (ego-browser), for
text mining under the user's institutional subscription. A deterministic script already tried each paper
and failed; your job is to handle irregular pages: institution sign-in entry points (choosing the
institution / "access through your institution" when the browser is already signed in), different download
buttons, PDF viewers and wrapper pages.

## Papers

```json
$papers_json
```

## How to use the browser

- Run browser scripts only with `ego-browser nodejs` (heredoc on stdin). The first line of every script
  MUST be `// paper: N` with the paper's number from the list above. Use only the TaskSpace
  `taskSpace("$space")`; it starts with Page `p1`, reuse it with `goto()`.
- Save a downloaded PDF only into that paper's `save_dir` (absolute path), for example with
  `download.saveAs(save_dir + "/paper.pdf")` after `page.waitForEvent("download")`, or
  `page.fetch(pdfUrl, { saveAs: save_dir + "/paper.pdf" })` for a same-site PDF link.
- The wrapper may answer that a publisher is paced ("slot opens in N s"): work on another paper and come back
  later, or stop. Never try to get around pacing.
- If the wrapper prints `STOP`, end immediately.

## Hard rules (the wrapper and a guard enforce most of them; a violation ends the batch)

1. Never interact with any verification: no CAPTCHA, no "verify you are human", no Cloudflare / Turnstile
   / PerimeterX / slider / checkbox challenge, no puzzle. If a page shows one, stop working on that paper
   and report `challenge`.
2. Never type, read or reveal credentials: no usernames, passwords, one-time codes, cookies or tokens. Do not
   fill any form field. If a page asks you to sign in, report `login_required` for that paper.
3. Visit only the `allowed_domains` of the paper you are working on (its DOI resolver and publisher
   sites). Do not search the web. Never use Sci-Hub, LibGen, Z-Library, Anna's Archive or any other
   unauthorized source, and no preprint mirrors outside the allowed domains.
4. Respect the download ledger: at most one paper of a publisher at a time; the same publisher is paced
   $gap_s–$gap_max_s seconds apart, and the daily caps are enforced by the wrapper.
5. Do not open, close, read or switch the user's own tabs or TaskSpaces. Use only "$space".
6. Do not run any other program: only `ego-browser nodejs`. No curl, wget, python, node, open, ssh.
7. At the very end close your TaskSpace: run `ego-browser nodejs` with the single line `// finish` as the
   script (the wrapper closes "$space" for you).

## Time budget

About $minutes minutes in total. Spend at most ~3 browser calls per paper; skip papers that do not work.

## Final answer

One line per paper: `N: saved | challenge | login_required | no_access | not_found | skipped`, nothing else.
Your report is informational only; the files are checked by code.
