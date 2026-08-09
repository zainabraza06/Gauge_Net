# Deployment

## Why not Vercel

Vercel hosts the frontend well and cannot host this app at all. Three
independent blockers, any one of which is fatal:

| Blocker | Detail |
|---|---|
| Bundle size | ~911 MB of dependencies and weights (torch alone is 514 MB installed) against a ~250 MB serverless function limit |
| Execution model | Gradio needs a persistent server with an SSE/WebSocket queue; Vercel functions are request-scoped and are torn down between calls |
| Duration cap | 10 s on Hobby, 60 s on Pro, against roughly a second per engine and up to 259 engines in one FD002 upload |

A CPU-only torch build does not close a 3.6× size gap, and no bundling trick
makes a request-scoped function hold a queue open. If a `vercel.app` URL is
specifically wanted, host the model elsewhere and put a thin frontend on
Vercel — see the last section.

## Hugging Face Spaces (recommended)

Free, purpose-built for Gradio, carries the 58 MB of weights natively, and has
no request timeout.

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create gaugenet-rul --type space --space_sdk gradio

git clone https://huggingface.co/spaces/<your-username>/gaugenet-rul
cd gaugenet-rul
```

Copy in what the Space needs — the serving code, the model package, the
weights, and the bundles:

```bash
REPO=/path/to/TurboEngine_RLU_Prediction

cp -r $REPO/model  $REPO/ablation  $REPO/app  $REPO/checkpoints  .
cp $REPO/deploy/space/space_app.py  $REPO/deploy/space/requirements.txt  .
cp $REPO/deploy/space/README.md     .          # overwrites the Space's stub
```

`checkpoints/` and `app/bundles/` are both required. The weights alone cannot
serve: the normalisation statistics, the healthy geometric reference and the
fleet median life were fitted on the training split and live in the bundles.

Files over 10 MB need LFS, which none of these are — the largest is a 3 MB
checkpoint — so a plain push works:

```bash
git add -A && git commit -m "GaugeNet RUL demo" && git push
```

The Space builds in a few minutes and serves at
`https://huggingface.co/spaces/<your-username>/gaugenet-rul`.

### Two things to know

**`space_app.py`, not `app.py`.** The serving code is a package called `app/`,
and a module named `app.py` beside a package named `app/` makes
`import app.predict` ambiguous. The Space README's `app_file:` field points at
`space_app.py` for exactly this reason — renaming it back will break the
import at startup.

**The hosted MC-dropout default is 25, not 100.** 100 passes reproduces the
reported numbers and costs about a second per engine on a CPU Space, so a full
259-engine FD002 upload takes minutes. The slider still reaches 100.

### Upgrading

A CPU Space is free and adequate. If uploads are large enough that latency
matters, the paid CPU upgrade helps more than a GPU would: the model is 746K
parameters and 26.7M MACs, so the bottleneck is the 100 sequential forward
passes per engine, not arithmetic throughput. Lowering MC passes is the
cheaper fix and is the first thing to try.

## Alternatives to Spaces

| Host | Fit |
|---|---|
| **Hugging Face Spaces** | best — free tier, Gradio-native, no timeout |
| **Render / Railway / Fly.io** | good — a persistent container, ~$5-7/month, needs a Dockerfile |
| **Modal / RunPod** | good for GPU-backed inference behind an API, pay per second |
| **Vercel / Netlify / Cloudflare Workers** | frontend only — see below |

## Keeping a Vercel URL

Vercel serves the page; the Space serves the model. The frontend calls the
Space over its HTTP API, so no PyTorch ever reaches Vercel.

```js
// app/api/predict/route.js  -- a Next.js route handler on Vercel
import { Client } from "@gradio/client";

export async function POST(request) {
  const form = await request.formData();
  const client = await Client.connect("<your-username>/gaugenet-rul");
  const result = await client.predict("/run", {
    file_obj: form.get("file"),
    subset: form.get("subset"),
    mc_passes: 25,
  });
  return Response.json(result.data);
}
```

Two caveats worth knowing before building on this:

- A free Space **sleeps after inactivity** and takes ~30 s to wake. The
  frontend needs to handle that first slow call rather than time out on it.
- Vercel's own duration cap still applies to the route handler, so it must not
  block on a long prediction. For large uploads, return the Space's job handle
  and poll, rather than awaiting the full result in one request.
