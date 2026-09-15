# OpenLR location references on Mapbox map data

Reading and decoding OpenLR location references against Mapbox map data: a
worked example of map-matching between two different road network providers.

> **Status.** The write-up below is complete. The reference implementation is
> not published yet. If you came here for working code, the issues are the
> place to say so, and it will move up.

## The problem

Two systems want to talk about the same stretch of road. One of them has
TomTom's map, the other has Mapbox's, and neither can send the other a road ID,
because those IDs only mean something inside the map that issued them. Road
networks are also not the same shape from one provider to the next: a single
carriageway in one may be a dual carriageway in another, a junction may be
modelled as one node or as four, and the geometry rarely agrees to the metre.

OpenLR is an open standard, originally from TomTom, for describing a location
on a road network without referring to any map's identifiers. Instead of an ID
it sends a small, map-agnostic description of the road itself, which the
receiver matches against whatever map it happens to hold.

## What is in a location reference

An OpenLR line location is a chain of location reference points. Each one
carries roughly:

| field | what it means |
|---|---|
| coordinate | latitude and longitude of the point |
| functional road class | how important the road is, motorway down to residential |
| form of way | carriageway type: motorway, roundabout, slip road and so on |
| bearing | the direction of travel leaving that point |
| distance to next | metres along the road to the following point |

Plus offsets at each end, for when the location starts or finishes part-way
along a road rather than at a junction.

The whole thing is compact enough to fit in a binary blob a few dozen bytes
long, which is the point: it travels well.

## Why decoding is the hard part

Encoding is mostly bookkeeping. Decoding is a search.

For each reference point the decoder finds candidate lines in its own map,
scored on how well the coordinate, bearing, road class and form of way agree.
None of them will agree exactly. Then it looks for a route between consecutive
candidates whose length is close to the distance the reference claims, and
prefers the combination where the whole chain is consistent rather than the one
where each point individually scored best.

That is where cross-provider matching gets interesting, and where a worked
example is more use than a specification:

- **Road class disagreement.** Providers classify the same road differently,
  often by one level. A strict match discards the right candidate.
- **Bearing near junctions.** Bearing is measured over a short distance from
  the point, so it is unstable exactly where reference points tend to sit.
- **Distance mismatch.** Different geometry means different lengths along what
  is nominally the same route. The tolerance has to absorb that without
  becoming so loose it accepts a parallel road.
- **Dual carriageways.** The two directions can be one line or two, and picking
  the wrong side silently reverses the location.

## What this repository is for

A small, readable example of the decode path against Mapbox data, with the
scoring made explicit rather than buried, so the failure modes above can be
seen rather than described.

## Reference

- [openlr.org](https://www.openlr.org/), the standard's home, for the format
  and the decoding rules.
- [tomtom-international/openlr](https://github.com/tomtom-international/openlr),
  the Apache-2.0 Java reference implementation, which is the clearest statement
  of what a conforming decoder does.
- [Mapbox Directions and Map Matching APIs](https://docs.mapbox.com/api/navigation/),
  the map side of the problem.

## Contributing

Issues and pull requests are welcome. If you are working on OpenLR against a
provider other than Mapbox, the failure modes are usually the same, and a note
about which ones bit you is genuinely useful.

## Licence

[Apache-2.0](LICENSE) — Yauhen Bichel

---

## Contributors

Thank you to everyone who has helped.

<!-- readme: contributors,bots/- -start -->
<p align="center">
  <a href="https://github.com/YauhenBichel" title="Yauhen Bichel" aria-label="Yauhen Bichel"><img src=".github/faces/YauhenBichel.svg" width="87" height="99" alt="Yauhen Bichel" /></a>
</p>
<!-- readme: contributors,bots/- -end -->

Filled from GitHub commits (bots omitted). Live demo: [readme-contributors](https://github.com/YauhenBichel/readme-contributors#live-demo).
