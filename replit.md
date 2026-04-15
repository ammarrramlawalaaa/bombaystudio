# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## Flask App — Bombay Studio

A standalone Python Flask web app at `flask-app/app.py`.

### Stack
- **Framework**: Flask 3.x
- **Face recognition**: `face_recognition` + `dlib`
- **Image processing**: `opencv-python`, `Pillow`, `numpy`
- **Database**: SQLite3 (`flask-app/faces.db`)
- **Frontend**: Jinja2 templates + Bootstrap 5
- **Photos stored**: `flask-app/uploads/`

### Routes
- `/` — Landing page
- `/admin` — Upload event photos & manage indexed photos (POST: process + store faces)
- `/admin/delete/<filename>` — Remove a photo and its face encodings
- `/guest` — Guest selfie upload form
- `/uploads/<filename>` — Serve stored photos

### Key Commands
- `cd flask-app && python3 app.py` — Run the Flask app (port 5000)

### How it works
1. Admin uploads photos → `face_recognition` extracts face encodings → stored as BLOBs in SQLite
2. Guest uploads selfie → encoding extracted → compared against all stored encodings
3. Matched photos returned as a gallery
