// Railway Infrastructure as Code for https://cfb.tzadek.ai (railway.json, "Config as Code", is
// deprecated: new services no longer read it). The Dockerfile at the repo root is detected
// automatically, so the only things declared here are what the dashboard would otherwise be
// clicked through: the GitHub source, Streamlit's health endpoint and the custom domain.
//
//   railway login && railway link      # once, to the project this should live in
//   railway config plan                # shows what would change, touches nothing
//   railway config apply
//
// CFBD_API_KEY is a secret: set it on the service in the Railway dashboard (Variables), not here.
import { defineRailway, github, project, service } from "railway/iac";

export default defineRailway(() => {
  const dashboard = service("cfb-ratings", {
    source: github("bpaikoff/BrandonCFBRatings", { branch: "master" }),
    healthcheck: "/_stcore/health",
    healthcheckTimeout: 120,
    domains: ["cfb.tzadek.ai"],
  });

  return project("cfb-ratings", {
    resources: [dashboard],
  });
});
