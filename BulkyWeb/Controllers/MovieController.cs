using Bulky.DataAccess.Data;
using Bulky.Models;
using Microsoft.AspNetCore.Mvc;
using System.Net.Http.Json;
using System.Text.Json;

namespace BulkyWeb.Controllers
{
    public class MovieController : Controller
    {
        private readonly ApplicationDbContext _db;
        private readonly IHttpClientFactory _httpClientFactory;
        private readonly IConfiguration _config;

        // The Python agent service uses snake_case JSON (FastAPI/Pydantic convention)
        private static readonly JsonSerializerOptions AgentJsonOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            PropertyNameCaseInsensitive = true
        };

        public MovieController(ApplicationDbContext db, IHttpClientFactory httpClientFactory, IConfiguration config)
        {
            _db = db;
            _httpClientFactory = httpClientFactory;
            _config = config;
        }

        public IActionResult Index()
        {
            // Fetch all movies from the database
            List<Movie> objMovieList = _db.Movies.ToList();
            return View(objMovieList);
        }

        public IActionResult Create()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Create(Movie obj)
        {
            // Custom Validation: Prevent creating a movie with the same title and description
            if (obj.Title != null && obj.Title.ToLower() == obj.Description?.ToLower())
            {
                ModelState.AddModelError("title", "The Description cannot exactly match the Title.");
            }

            if (ModelState.IsValid)
            {
                _db.Movies.Add(obj);
                _db.SaveChanges();
                var indexed = await TryIndexAsync(obj);
                TempData["success"] = indexed
                    ? "Movie created successfully"
                    : "Movie created, but it could not be added to the script catalog";
                return RedirectToAction("Index");
            }

            return View(obj);
        }

        public IActionResult Edit(int? id)
        {
            if (id == null || id == 0) return NotFound();

            Movie? movieFromDb = _db.Movies.Find(id);

            if (movieFromDb == null) return NotFound();

            return View(movieFromDb);
        }

        [HttpPost]
        public async Task<IActionResult> Edit(Movie obj)
        {
            if (ModelState.IsValid)
            {
                _db.Movies.Update(obj);
                _db.SaveChanges();
                var indexed = await TryIndexAsync(obj);
                TempData["success"] = indexed
                    ? "Movie updated successfully"
                    : "Movie updated, but the script catalog could not be refreshed";
                return RedirectToAction("Index");
            }

            return View(obj);
        }

        public IActionResult Delete(int? id)
        {
            if (id == null || id == 0) return NotFound();

            Movie? movieFromDb = _db.Movies.Find(id);

            if (movieFromDb == null) return NotFound();

            return View(movieFromDb);
        }

        [HttpPost, ActionName("Delete")]
        public async Task<IActionResult> DeletePOST(int? id)
        {
            Movie? obj = _db.Movies.Find(id);
            if (obj == null) return NotFound();

            _db.Movies.Remove(obj);
            _db.SaveChanges();
            await TryRemoveIndexAsync(obj.Id);
            TempData["success"] = "Movie deleted successfully";
            return RedirectToAction("Index");
        }

        [HttpPost]
        public async Task<IActionResult> GenerateAiScript([FromBody] ScriptRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.Prompt))
            {
                return Json(new { success = false, message = "Please describe what should happen in the scene." });
            }

            var client = _httpClientFactory.CreateClient("ScriptAgent");

            // Sends the movie details to the CrewAI agent service (ScriptAgentService/)
            using var message = new HttpRequestMessage(HttpMethod.Post, "generate")
            {
                Content = JsonContent.Create(
                    new { request.Title, request.Genre, request.Description, request.Prompt },
                    options: AgentJsonOptions)
            };
            message.Headers.Add("X-Api-Key", _config["SERVICE_API_KEY"]);

            try
            {
                using var response = await client.SendAsync(message, HttpContext.RequestAborted);

                if (!response.IsSuccessStatusCode)
                {
                    var errorMsg = await response.Content.ReadAsStringAsync();
                    return Json(new { success = false, message = $"Agent service error ({(int)response.StatusCode}): {errorMsg}" });
                }

                var result = await response.Content.ReadFromJsonAsync<AgentScriptResponse>(AgentJsonOptions);

                return Json(new
                {
                    success = true,
                    script = result?.Script,
                    sources = result?.Sources ?? new List<CatalogSource>(),
                    contextTokens = result?.ContextTokens ?? 0
                });
            }
            catch (HttpRequestException)
            {
                return Json(new
                {
                    success = false,
                    message = $"Could not reach the script agent service at {client.BaseAddress}. Is it running?"
                });
            }
            catch (TaskCanceledException) when (!HttpContext.RequestAborted.IsCancellationRequested)
            {
                return Json(new { success = false, message = "The agents took too long to respond. Please try again." });
            }
        }

        // Helper class for the data coming from JavaScript
        public class ScriptRequest
        {
            public string? Prompt { get; set; }
            public string? Title { get; set; }
            public string? Genre { get; set; }
            public string? Description { get; set; }
        }

        public class AgentScriptResponse
        {
            public string Script { get; set; } = "";
            public List<CatalogSource> Sources { get; set; } = new();
            public int ContextTokens { get; set; }
        }

        public class CatalogSource
        {
            public int MovieId { get; set; }
            public string Title { get; set; } = "";
            public string Genre { get; set; } = "";
            public double Score { get; set; }
        }

        // Best-effort catalog updates. Saving the movie must succeed even when
        // the agent service or the vector database is down.
        private async Task<bool> TryIndexAsync(Movie movie)
        {
            using var message = new HttpRequestMessage(HttpMethod.Post, "index")
            {
                Content = JsonContent.Create(
                    new { movie.Id, movie.Title, movie.Genre, movie.Description, Script = movie.MovieScript },
                    options: AgentJsonOptions)
            };
            return await SendAgentAsync(message);
        }

        private async Task<bool> TryRemoveIndexAsync(int movieId)
        {
            using var message = new HttpRequestMessage(HttpMethod.Delete, $"index/{movieId}");
            return await SendAgentAsync(message);
        }

        private async Task<bool> SendAgentAsync(HttpRequestMessage message)
        {
            var client = _httpClientFactory.CreateClient("ScriptAgent");
            message.Headers.Add("X-Api-Key", _config["SERVICE_API_KEY"]);
            try
            {
                using var response = await client.SendAsync(message);
                return response.IsSuccessStatusCode;
            }
            catch (HttpRequestException)
            {
                return false;
            }
            catch (TaskCanceledException)
            {
                return false;
            }
        }
    }
}
