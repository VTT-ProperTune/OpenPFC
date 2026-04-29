<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Image prompts for OpenPFC artwork

## Project briefing (read this first)

**OpenPFC** is an **open-source software framework** (C++17, **AGPL-3.0-or-later**) for **large-scale, three-dimensional phase field crystal (PFC) simulations** in **materials science and engineering**. It is developed in a **research** context (e.g. association with **VTT Technical Research Centre of Finland Ltd** and collaborators) and is meant for **serious computational materials work**, not consumer apps or games. The project’s visuals should feel like **peer-reviewed science**, **HPC engineering**, and **crystalline physics**—not fantasy, not stock “tech” clichés unrelated to solids.

### What phase field crystal (PFC) means (for imagery)

- **PFC** is a **semi-atomistic** modeling approach: it captures **atomic-scale structure** of **crystals** (lattices, defects) while evolving on **longer, diffusive time scales**—so you can show **ordered lattices**, **grains**, **grain boundaries**, **dislocations**, **stacking faults**, **voids**, and related **microstructure** in a **physically grounded** way.
- Typical **phenomena** the science speaks about include **solidification** and **microstructure evolution**, **elastic–plastic** response, **epitaxial growth**, **phase transitions**, and **defect formation**—images that suggest **metallic or crystalline micrographs**, **simulation renders**, or **clean schematics** fit better than unrelated domains (fluids-only art, biological cells, generic molecules, gemstones, blockchain “crystal” metaphors).

### What OpenPFC does computationally (visual metaphors that fit)

- Simulations run on **large structured 3D grids**; the numerical stack is **spectral / FFT-heavy** (frequency-domain ideas are fair game as **subtle** background motifs—rings, symmetric lobes, grids—**not** as fake equations or unreadable labels).
- Runs are **parallelized** with **MPI**; the domain is **decomposed** across ranks; **halo / ghost-cell exchange** between neighboring blocks is part of the story—**abstract** “tiles exchanging thin boundary layers” or synchronized light links between blocks can suggest **parallelism** without pretending to be a screenshot.
- The code is designed to scale from **workstations** to **supercomputers**; **CPU** and optional **GPU** FFT backends matter—**calm** “many nodes / throughput / scale” abstractions work; **avoid** branding specific vendors, supercomputer names as logos, or fake job IDs.

### Tone and aesthetic that work for us

- Prefer **scientific visualization**, **microscopy-like clarity**, **muted metallics**, **cool neutrals**, **restrained accent** (one amber/teal/cyan line—not rainbow gradients).
- Favor **readability for UI**: **negative space** for titles, **no busy noise** in the center of banners, **contrast** that matches **dark-mode** or **light-mode** docs (see below).
- **Authority**: clean, **institutional**, **engineering**—like a **materials journal** cover or a **respectable HPC project**—not startup hype, not neon cyberpunk unless explicitly requested.

### What to avoid unless the user asks for it

- **Invented text, logos, or watermarks** (often gibberish)—explicitly exclude them in prompts.
- **Fantasy metal**, **magic crystals**, **jewelry**, **swords**, **generic blockchain “crystal”** imagery.
- **Misleading science**: random organic blobs, **protein** or **medical** motifs, **unrelated** chemistry glassware, **stock “data science”** dashboards with fake charts.
- **Overly literal** “screenshots” of code or terminals as **hero art** (cluttered, illegible)—prefer **abstract** representations of **simulation** and **scale**.

### How to use this document

The sections below give **copy-paste prompts** for heroes, headers, banners, logos, and other assets, including **dark UI** vs **light UI** variants. Adjust aspect ratios to your tool (e.g. hero **16:9** or **21:9**, social **1.91:1** or **1200×630**).

**Tips:** add *“no text, no watermark, no logo”* if the model invents illegible lettering; add *“scientific visualization style, physically plausible”* to steer away from fantasy metal. For **logos**, prefer **vector export** (Illustrator, Inkscape) or trace clean AI shapes—generators rarely output production-ready SVG.

---

## Theme: dark UI vs light UI

Use **dark UI** prompts when the artwork sits behind **light typography** (white, pale cyan, soft gray text) or on **dark-themed** docs and sites: favor **deep backgrounds** (#0a0f18–#1a2332), **luminous** lattice edges, **volumetric glow**, **high contrast** accents.

Use **light UI** prompts when the artwork sits behind **dark text** or on **print / academic PDFs**: favor **off-white or cool gray** bases (#f4f6f8–#e8ecf0), **soft shadows**, **muted metallics**, **ink-navy** lines, **low glare**—avoid pure #ffffff if the model blows highlights.

You can append **global tokens** to any prompt:

- **Dark UI token:** *dark mode UI background, deep charcoal and navy, subtle bloom on crystal edges, high contrast, generous empty area for white headline text*
- **Light UI token:** *light mode UI background, soft cool-gray paper texture, subtle grain, restrained contrast, generous empty area for dark headline text*

---

## Hero images (landing / homepage)

Full-bleed backgrounds; aim for **negative space** on one side for headline + CTA.

### Dark UI

1. **Crystalline front** — *Ultra-wide cinematic hero, abstract 3D phase-field crystal microstructure: polycrystalline metal grains meeting at grain boundaries, subtle dislocation lines and stacking faults visible as faint defects in the lattice, volumetric lighting from upper left, deep teal and graphite palette with a single warm amber accent on grain edges, shallow depth of field, scientific visualization aesthetic, 8k detail, no text.*

2. **Solidification wave** — *Epic hero image: rapid solidification front advancing through undercooled melt, semi-abstract transition from disordered fluid to ordered crystalline lattice, FFT-inspired concentric frequency rings faintly overlaid as a subtle graphic motif (not text), cool blue-gray to silver gradient, dramatic but clean, suitable for a materials-science software product, no logos.*

3. **Exascale abstraction** — *Minimal futuristic hero: infinite grid of dimly lit compute nodes fading into darkness, ghostly transparent 3D crystal lattice superimposed in the foreground representing simulation domain decomposition, MPI halo exchange suggested by thin synchronized light pulses between blocks, dark background, electric cyan and white highlights, high-end tech keynote style, vast negative space on the right for typography.*

### Light UI

1. **Crystalline front** — *Ultra-wide hero, same phase-field polycrystal subject as a scientific viz but rendered for light UI: soft off-white and pale blue-gray background, microstructure in muted steel and sage with gentle contact shadows, no harsh blacks, airy negative space on the right, crisp but low-contrast, no text.*

2. **Solidification wave** — *Epic hero for light theme: solidification front as soft watercolor-like transition from misty disordered region to faint lattice lines, pearlescent white and ice-blue palette, very subtle FFT ring motif as barely visible pencil-thin circles, calm institutional look, no logos.*

3. **Exascale abstraction** — *Light-mode hero: isometric wireframe lattice and faint server-grid metaphor on pale cool-gray backdrop, thin hairline cyan strokes suggesting data movement, almost flat with soft drop shadow, plenty of whitespace for dark sans-serif title, minimalist.*

---

## Header images (docs, blog, narrow strips)

Shorter vertical depth than heroes; strong **horizontal** composition.

### Dark UI

1. **Lattice strip** — *Wide thin banner crop, repeating hexagonal crystal lattice viewed obliquely, soft depth of field, muted steel blue and soft white, gentle vignette, scientific illustration quality, seamless feel left-to-right.*

2. **FFT spectrum ribbon** — *Horizontal header: abstract Fourier domain visualization—radial spectrum with soft glowing rings and symmetric lobes, dark navy background, thin lines, no numbers or axes labels, calm technical mood.*

3. **Tungsten-inspired surface** — *Narrow header: nanoscale metallic surface with faceted grains and a few lattice defects as subtle darker threads, side lighting, monochrome with one muted gold rim light, macro photography meets simulation render.*

### Light UI

1. **Lattice strip** — *Wide thin banner, same hex lattice motif but light theme: white to pale gray gradient background, lattice in soft graphite lines with faint blue shadow, delicate vignette, seamless horizontal tiling feel, no text.*

2. **FFT spectrum ribbon** — *Horizontal header: Fourier spectrum as thin ink-navy and periwinkle lines on off-white, very subtle lavender lobes, looks like a journal figure header, no labels or digits.*

3. **Tungsten-inspired surface** — *Narrow header: bright metallic microstructure with soft daylight-style key light, silver and pale gold, defects as slightly darker brushed streaks, high-key photography look, low contrast.*

---

## Banners (GitHub social preview, conference slides, repo)

Often **~2:1** or **1280×640**; keep focal content **center-safe** (GitHub crops).

### Dark UI

1. **Open science + HPC** — *Banner: split composition—left side abstract crystal growth, right side stylized server racks / supercomputer silhouette dissolving into particles, bridge in the middle is a flowing field line, colors deep blue and silver, AGPL open-source vibe without showing any real trademark, no text.*

2. **Materials pipeline** — *Wide conference banner: left to right storyboard—raw disordered atoms → coarsening microstructure → ordered polycrystal, flat design meets soft 3D shading, limited palette (slate, cyan, graphite), plenty of empty center band for title overlay later.*

3. **Performance curve implied** — *Abstract banner: rising translucent layers like stacked spectral planes or memory slabs, subtle upward motion blur suggesting scalability, dark background, green-to-cyan gradient accents suggesting “efficient scaling”, minimalist, no charts or digits.*

### Light UI

1. **Open science + HPC** — *Banner for light UI: same split crystal / HPC story but pastel backgrounds—left pale mint-gray, right soft cloud gray—silhouettes in steel blue, thin white card-style divider in the middle, flat illustration, center band left empty for title, no text in image.*

2. **Materials pipeline** — *Wide banner: same left-to-right evolution in soft editorial illustration style, paper-white background, grains as light blue-gray shapes with fine outlines, generous whitespace in the middle third.*

3. **Performance curve implied** — *Abstract light banner: stacked translucent white layers with soft blue shadows suggesting depth and scale, subtle upward diagonal composition, pale gray background, mint or teal accent only on top layer edge, no digits.*

---

## Logos and marks

Use prompts as **inspiration**; refine in vector tools. Prefer **simple geometry** for small sizes. **Dark UI** = marks intended to sit on **dark** backgrounds (often light-filled strokes). **Light UI** = marks on **light** backgrounds (dark strokes).

### Dark UI (light-on-dark marks)

1. **Monogram “PFC”** — *Flat vector logo concept: stylized letters P F C integrated into a single compact mark suggesting a crystal unit cell and FFT grid lines, **light gray or white** on **transparent or dark circular badge**, geometric, modern tech, thick clean strokes, no gradients or fine hairlines.*

2. **Lattice hex + wave** — *Minimal logo mark: regular hexagon made of six small spheres at vertices with one subtle sine wave crossing the center (spectral methods), **pale cyan or white** on **deep navy disk**, Swiss design discipline, app-icon friendly.*

3. **Open frame** — *Wordmark-free symbol only: open cubic lattice corner fragment forming an incomplete box (open source), isometric, rounded joints, **frosted silver lines** on **charcoal** background mockup, friendly engineering aesthetic, 64×64 px clarity.*

### Light UI (dark-on-light marks)

1. **Monogram “PFC”** — *Same PFC crystal-grid monogram concept, **navy and graphite** strokes on **white or very light gray** square, slightly heavier weight for small-size legibility, no fake 3D bevel.*

2. **Lattice hex + wave** — *Same hex + sine motif, **single ink-navy** (#1c2a3a) on **transparent**, optional very subtle cool-gray circle behind, print-safe.*

3. **Open frame** — *Same open lattice box symbol, **two-tone navy and steel blue** on **off-white**, slightly rounded corners on the outer icon bounding shape, works on documentation headers.*

---

## Other useful artwork

### Social / Open Graph (square or 1.91:1)

**Dark UI**

1. *OG image: centered abstract 3D crystal cluster with soft bloom, dark vignette, generous margin, mood “peer-reviewed materials software”, no text—reserve corners for later overlay in HTML.*

2. *Square social card: Voronoi-like grain structure in deep metallic blues and purples on near-black, soft bloom, cinematic thumbnail.*

**Light UI**

1. *OG image: same crystal cluster subject, bright center, soft gray-white surround, very subtle edge vignette only, calm academic poster feel, no text.*

2. *Square social card: top-down grain map in pastel blue-gray and cream, soft shadows, editorial illustration, plenty of margin.*

### Icons & small UI metaphors

**Dark UI**

1. *App icon: simplified 3D cube of atoms with one highlighted bond, flat shading, **glowing accent bond**, dark rounded-square background, high contrast, readable at 32 px.*

2. *MPI halo exchange: two tiles exchanging glowing slab on **dark charcoal** field, isometric, neon-cyan accent.*

**Light UI**

1. *App icon: same atom cube, **navy lines** on **light gray** rounded square, matte finish, readable at 32 px.*

2. *MPI halo exchange: same metaphor on **white** with **slate** tiles and **teal** halo line, flat design.*

### Presentation / title backgrounds

**Dark UI**

1. *16:9 slide master: soft blurred microstructure bokeh, very dark blue-gray, a few sharp lattice edges in foreground only at bottom third, large empty upper area for white text.*

2. *16:9: abstract “frequency space” wallpaper—diffuse glowing nodes on a grid, suggests HeFFTe/FFT without literal UI, restrained color.*

**Light UI**

1. *16:9 slide master: pale gray-white with subtle paper grain, faint blurred lattice texture confined to lower 25%, large clean upper area for dark text.*

2. *16:9: light gray background with sparse **ink** grid nodes and whisper-thin connecting lines, suggests spectral mesh, almost invisible until fullscreen.*

### Merch / sticker (optional)

**Dark UI** — *Sticker on dark garment: dislocation cartoon on **midnight blue** chunk, thick outlines, **neon** accent line, white outer border.*

**Light UI** — *Sticker for light surface: same cute dislocation crystal, **navy outlines** on **warm white** fill, vintage science textbook vibe.*

---

## Quick reference (themes from the project)

| Theme | Prompt keywords |
|--------|------------------|
| Physics | Phase field crystal, polycrystal, grain boundaries, dislocations, solidification, epitaxy |
| Methods | Spectral methods, FFT, k-space, periodic domain |
| Computing | MPI, domain decomposition, halo exchange, CPU/GPU backends, scalability |
| Identity | Open source (AGPL), C++17, materials science, VTT / research software |

When generating **series** (hero + banner + icon), lock a **palette** and **style token** across prompts (e.g. *“same palette: #0b1f33, #3d7ea6, #c9a227”*) so assets match. For **dark + light** pairs of the same motif, reuse the **same subject nouns** (e.g. “hex lattice header”) and only swap the **Dark UI token** / **Light UI token** paragraphs above.
