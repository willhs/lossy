---
name: productionise
description: "Guide for deploying applications to production. This skill should be used when setting up CI/CD pipelines, creating Docker infrastructure, preparing for first production deployment, or adding deployment automation to existing projects."
---

# Productionise

This skill provides guidance for setting up production deployment infrastructure.

## When to Use

- Setting up CI/CD pipelines for a project
- Creating Docker infrastructure for deployment
- Preparing an application for first production deployment
- Adding deployment automation to existing projects
- When user mentions "productionise", "deploy", "CI/CD", or "Docker setup"

## Pre-flight Checklist

Before setting up deployment infrastructure, verify:

1. **Health check endpoint exists** - Required for container orchestration
2. **Environment variables documented** - Know what config the app needs
3. **Secrets identified** - Ensure sensitive values are excluded from git
4. **Build process works locally** - `npm run build` or equivalent succeeds
5. **Ports documented** - Know what ports the app uses

## Deployment Architecture Pattern

The recommended pattern uses GitHub Actions + Watchtower for zero-touch deployments:

```
Push to main → GitHub Actions builds images → Pushes to GHCR → Watchtower on VM auto-deploys
```

**Benefits:**
- Zero-downtime deployments
- No manual intervention after merge
- Simple operational model
- Works for personal projects and small teams

## Files to Create

### For Monorepo with API + Web

| File | Purpose |
|------|---------|
| `apps/api/Dockerfile` | Multi-stage Node Alpine build for API |
| `apps/web/Dockerfile` | Multi-stage build with nginx:alpine for SPA |
| `apps/web/nginx.conf` | Reverse proxy + SPA fallback |
| `apps/api/.dockerignore` | Exclude node_modules, dist, db files |
| `apps/web/.dockerignore` | Exclude node_modules, dist |
| `docker-compose.yml` | API, Web, Cloudflare Tunnel, Watchtower |
| `.github/workflows/deploy.yml` | Build & push images to GHCR |
| `docs/design/adr/XXX-deployment-architecture.md` | Document decisions |

### For Single App

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build |
| `.dockerignore` | Exclude dev files |
| `docker-compose.yml` | App, Cloudflare Tunnel, Watchtower |
| `.github/workflows/deploy.yml` | Build & push to GHCR |

## Implementation Steps

When productionising a project:

1. **Create GitHub repo** if not exists: `gh repo create username/project --private --source=. --push`
2. **Create .dockerignore files** for each app
3. **Create nginx.conf** if serving a SPA
4. **Create Dockerfiles** for each app
5. **Create docker-compose.yml** with all services
6. **Create GitHub Actions workflow**
7. **Update .gitignore** to exclude `/data/` (for docker volumes)
8. **Create ADR** documenting deployment decisions
9. **Commit and push** to trigger first CI run
10. **Verify** images appear in GHCR

## Verification Checklist

After setup, verify:

- [ ] GitHub Actions builds successfully
- [ ] Docker images appear in GHCR
- [ ] `docker-compose up` runs locally
- [ ] API responds at `/health`
- [ ] Web serves SPA correctly
- [ ] API proxy works through nginx
